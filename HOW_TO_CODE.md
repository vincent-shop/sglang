Chapter 4 Modules Should Be Deep
A module usually is any unit of code that has an interface and an implementation. The interface consists of everything that a developer working in a different module must know in order to use the given module. The implementation consists of the code that carries out the promises made by the interface.

The best modules are those whose interfaces are much simpler than their implementations. Such modules have two advantages. First, a simple interface minimizes the complexity that a module imposes on the rest of the system. Second, a simpler interface does not change when the module is modified or evolved.

The interface presents a simplified view of the module’s functionality by omitting the unimportant information. However, how to “simplified” the “unimportant” information is crucial. Considering a file system, the mechanism for choosing which blocks on a storage device to use for the data in a given file can be omitted, but when data is written through to storage must be visible to some applicats like databases so they can ensure that data will be preserved after system crashes.

Deep modules or Shallow modules
Deep modules: they provide powerful functionality yet have simple interfaces (the following UNIX I/O), many implementation details are hidden.
int open(const char* path, int flags, mode_t permissions);
ssize_t read(int fd, void* buffer, size_t count);
ssize_t write(int fd, const void* buffer, size_t count);
off_t lseek(int fd, off_t offset, int referencePosition);
int close(int fd);
Shallow modules: their interface is relatively complex in comparison to the functionality that it provides (the following java function).
private void addNullValueForAttribute(String attribute) {
    data.put(attribute, null);
}
The value of deep classes is not widely appreciated today. The conventional wisdom in programming is that classes should be small, not deep. Some conventional principles are often conveyed like “One class just does one simple job”, “Any method longer than N lines should be divided into multiple methods”. However, a large numbers of shallow classes and methods will add overall system complexity significantly.

Examples
One of the most visible examples of classitis today is the Java class library. For example, to open a file in order to read serialized objects from it, you must create three different objects.The people who support this design might argue that it’s better to keep buffering separate, so people can choose whether or not to use it. Providing choice is good, but interfaces should be designed to make the common case as simple as possible. Almost every user of file I/O will want buffering, so it should be provided by default. For those few situations where buffering is not desirable, the library can provide a mechanism to disable it.

FileInputStream fileStream = new FileInputStream(fileName);

BufferedInputStream bufferedStream = new BufferedInputStream(fileStream);

ObjectInputStream objectStream = new ObjectInputStream(bufferedStream);
Chapter 5 Information Hiding and Leakage
Information hiding and deep modules are closely related. Essentially, hiding means no dependency and leakage means creating unnecessary dependency.

The basic idea of information hiding is that each module should encapsulate a few pieces of knowledge, which represent design decisions. The knowledge is embedded in the module’s implementation but does not appear in its interface, so it is not visible to other modules. Information hiding reduces complexity in two ways. First, it simplifies the interface to a module. The other modules doesn’t need to and will never know the hidden information. Second, information hiding makes it easier to evolve the system. If a piece of information is hidden, there are no dependencies on that information outside the module containing the information, so a design change related to that information will affect only the one module.

Information leakage occurs when a design decision is reflected in multiple modules. This creates a dependency between the modules: any change to that design decision will require changes to all of the involved modules. Information leakage is one of the most important red flags in software design. When information leakage happened, try you best to avoid it.

Temporal decomposition
In temporal decomposition, the structure of a system corresponds to the time order in which operations will occur. Specifically, execution order is reflected in the code structure: operations that happen at different times are in different methods or classes. The problem is if the same knowledge is used at different points in execution, it gets involved in multiple places, resulting in information leakage.

Examples
When designing a HTTP server, you may use an object of type HTTPRequest to hold the parsed HTTP request, and the HTTPRequest class had a single method like the following one to return parameters:

public Map<String, String> getParams() {
    return this.params;
}
Rather than returning a single parameter, the method returns a reference to the Map used internally to store all of the parameters. This method is shallow, and it exposes the internal representation used by the HTTPRequest class to store parameters, which bring three issues.

Any change to that representation will result in a change to the interface, which will require modifications to all callers.

This approach also makes more work for callers: a caller must first invoke getParams, then it must call another method to retrieve a specific parameter from the Map.

A underlying but important requirement is that callers must realize that they should not modify the Map returned by getParams, since that will affect the internal state of the HTTPRequest.

A better interface for retrieving parameter values:

public String getParameter(String name) { ... }

public int getIntParameter(String name) { ... }
Chapter 6 General-Purpose Modules are Deeper
Question: When designing a new module, whether to implement it in a general-purpose or special-purpose fashion.

General-purpose: implementing a mechanism that can be used to address a broad range of problems not just the ones that are important today may find unanticipated uses in the future, thereby saving time.

Special-purpose: it’s hard to predict the future needs of a software system, so a general-purpose solution might include facilities that are never actually needed. Specializing and optimizing the module for the way you plan to use it today should bring additional benefits.

Author’s Answer: the sweet spot is to make an easy-to-use interface, but implement the module based on the current needs. So, the interface is general-purpose, implementation is special-purpose.

Example
Considering to design a GUI text editors, the editors had to display a file and allow users to point, click, and type to edit the file. So, if a user of the editor typed the backspace key, the editor deleted the character immediately to the left of the cursor; if the user typed the delete key, the editor deleted the character immediately to the right of the cursor.

Knowing this, a special-purpose way of implementation is shown as follows,

void backspace(Cursor cursor);

void delete(Cursor cursor);

void deleteSelection(Selection selection);
backspace and delete take the cursor position as its argument; a special type Cursor represents this position. The editor also had to support a selection that could be copied or deleted. So, defining a Selection class and passing an object of this class to the text class during deletions.

However, this implementation ended up with a large number of shallow methods, each of which was only suitable for one user interface operation. Many of the methods, such as delete, were only invoked in a single place. As a result, a developer working on the user interface had to learn about a large number of methods for the text class.

A better approach is to make the text class more generic. Its API should be defined only in terms of basic text features, without reflecting the higher-level operations that will be implemented with it.

void insert(Position position, String newText);

void delete(Position start, Position end);

Position changePosition(Position position, int numChars);

// delete method
text.delete(cursor, text.changePosition(cursor, 1));

// backspace method
text.delete(text.changePosition(cursor, -1), cursor);
Thus, we only have a general delete API, but it can support various operations like delete, backspace, and delete selection. Also, the code is much easier to read and understand.
