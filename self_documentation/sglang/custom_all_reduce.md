# SGL-custom all reduce

## Simple version

> [!IMPORTANT]
> Think of this as “one GPU reaches straight into the others, sums values, and writes them back” without any NCCL bookkeeping.

Directly write to one / the other buffer over NVLINK - does not use NCCL at all.
A signal struct tells when peers are ready before / after reduction
Two CU kernels are doing the most work:

- One for small payload for single sum, and there's also a two-stage reduce-scatter + all gather (becomes all-reduce) when you concat both of them. Notice the order of the difference.
- The c++ wrapper makes it friendly to use in torch. And cuda ipc makes each GPU see peer memory(so it's still cuda-graph compatible)
- Why does this work...?
CUDA lets the peer gpu issue normal global load/store once IPC/nvlink is enabled, and the custom barrier + bit packed load = low latency

## Quick bit packing primer
Every int is just a row of bits.
So several values share one word instead of using seperate variables.
I.e if the value does not take up the whole bit space, then put them.
So for each field, you put the bits you want in the same space it takes to just take one var.
In advance, you must decide:
- How manny bits each field needs
- Move the value in slots with left shirt, combine with OR. If you want to read, just reverse using mask + shift.
- Packing reduces the mem footprint and improve cache bandwith (less bytes)
But you need to stay within the bit width you decide in the original contract, and handle the signed / unsigned values. And manually pack / unpack using your own helpers, etc.

```cpp
// Stores two 16-bit unsigned ints inside one 32-bit word.
inline constexpr uint32_t pack_u16(uint16_t hi, uint16_t lo) {
  return (uint32_t(hi) << 16) | uint32_t(lo);
}

inline constexpr uint16_t unpack_hi(uint32_t word) {
  return static_cast<uint16_t>(word >> 16);
}

inline constexpr uint16_t unpack_lo(uint32_t word) {
  return static_cast<uint16_t>(word & 0xFFFF);
}
```

# From scratch
More detailed version.

---

### Step 1 – Setup

1. Setup
Each rank gets ptr to all peers signal buffer + slot to RankData, which is data array that holds peers' pointers for each registered tensor.
Then create the customallreduce obj, and share the IPC handle for signal buffer.
Then call the register_buffer once it gather peer tensor ptrs. Cache then upload to GPU
#TODO: this is not clear.
Clarification: `init_custom_ar` gives every rank the shared signal buffers and a device array that will later hold peer pointers. After the ranks trade tensor addresses, `register_buffer` copies that pointer table to the GPU. Once this is done, the kernels always know where to read each peer’s data and the host never has to intervene again.

> [!NOTE]
> After setup, every allreduce call can launch straight away because the peer pointer table already lives on the device.

### Step 2 – Integration with Graph Capture

2. Integration with Graph Capture
When cuda graph recorded, code doesn't know peer ptrs yet. So instead just reserve slots in advance and after capture, fill the slots (so it doesn't change the kernel arsg)
TODO: this is not clear.
During stream capture the kernel arguments must stay fixed, so the code just records which buffers will be part of the allreduce and postpones filling in their peer pointers. After the capture finishes and the real pointers are known, `register_graph_buffers` writes them into the reserved slots. Replays now see exactly the same argument layout that was captured.

> [!TIP]
> Capture: remember the addresses; Replay: patch them in; kernels never see the difference.

### Step 3 – Sync primitive

3. Sync primitive
Multi gpu barrier increment the local counter, then write into peer counter. Then wait until peer has the same value.
TODO: how this works not clear.
Each block keeps its own counter. When it reaches the barrier it bumps the counter, writes that value into the matching location on every peer, and spins until it sees the same number written back. Two alternating counter slots prevent a fast peer from stomping on the value a slower peer is still watching, and the optional fence makes sure data written before the barrier is visible everywhere once the wait finishes.

> [!CAUTION]
> No peer moves on until it sees its own counter mirrored back, so memory updates that happened before the barrier are safe to read.

### Step 4 – Packed Math

4. Packed Math
packed_t -> 128 bit load / store. So each last upcast to float32, then accumulate in same precision across all GPUs, then it's downcast. So you can apply the transaction everywhere and reduce the loss of precision from downcast.
TODO: why not downcast before accumulate?
The math uses 128-bit loads to pull several values at once. Each slice is promoted to float32, added across GPUs in that higher precision, and only then converted back to half or bfloat16. If you downcasted sooner, every partial sum would pick up extra rounding error.

> [!INFO]
> Wide loads make memory happy; float32 accumulation keeps the totals numerically stable.

### Step 5 – Kernel choice

5. Kernel choice [1/2 or 2/2]
Check the world size(TODO: what this mean)
Then check message size(bytes)
Then check for NVLINK full topology. Run single barrier reduction(TODO: find out what this means)
Otherwise reduce-scatter into per-rank scratch space stored after each `Signal`
Then allgather every piece back together.
Here `world_size_` just means how many GPUs take part. Combined with the message size and whether every pair of GPUs has NVLink, it picks the launch path. The light path (single-stage) does one round of “sum everyone, write results, synchronize”. The heavy path (two-stage) first has each rank sum the slice it owns, drops those partial sums into scratch space, fences, then copies the slices back so everybody ends up with the full result.

> [!SUMMARY]
> Small or sparse jobs use the single-stage kernel; larger or denser jobs pay one reduce-scatter/allgather cycle for better throughput.
