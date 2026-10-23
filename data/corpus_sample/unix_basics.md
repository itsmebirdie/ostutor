# File permissions and chmod

Every file on a Unix system has an owner, a group, and three sets of permission bits: read (r), write (w) and execute (x) for the owner, the group and everyone else. The command `ls -l` prints them as a string such as -rwxr-xr--.

The `chmod` command changes these bits. In octal form each digit is the sum of read = 4, write = 2 and execute = 1, written in the order owner, group, others. So `chmod 755 script.sh` gives the owner read, write and execute (7), and the group and others read and execute (5). The symbolic form `chmod u+x file` adds execute permission for the owner only.

For a directory, the execute bit means permission to enter it and access names inside it, while the read bit means permission to list its contents.

# Pipes and redirection

A pipe, written with the vertical bar, connects the standard output of one command to the standard input of the next, so `ls | wc -l` counts the entries in a directory. Redirection sends a stream to a file instead: `>` overwrites the file, `>>` appends to it, and `<` reads standard input from a file. Standard error is file descriptor 2 and can be redirected separately with `2>`.

The `grep` command prints the lines of its input that match a pattern. For example `grep -n error log.txt` prints matching lines together with their line numbers.

# Processes: fork, exec and wait

The `fork` system call creates a new process that is a copy of the caller. It returns the child's process id in the parent and zero in the child. The `exec` family replaces the current process image with a new program, keeping the same process id. A shell runs a command by forking, then having the child call exec while the parent calls `wait` to collect the child's exit status.

A child that has exited but whose parent has not yet called wait is a zombie process. The `ps` command lists processes and `kill` sends a signal to a process; by default it sends SIGTERM, which asks the process to terminate, while SIGKILL cannot be caught or ignored.

# Files, inodes and paths

On a Unix file system the metadata of a file, such as its size, owner, permissions and the location of its data blocks, is stored in an inode. A directory is a file that maps names to inode numbers. A hard link is another name for the same inode, whereas a symbolic link is a small file containing a path.

An absolute path starts at the root directory with a slash, such as /home/user/notes.txt. A relative path is resolved from the current working directory, which `pwd` prints and `cd` changes.

# System calls

A system call is the controlled entry point through which a user program asks the kernel for a service such as opening a file, creating a process or allocating memory. The processor switches from user mode to kernel mode when the call is made, and back again when it returns. Common examples are `open`, `read`, `write`, `close`, `fork`, `exec` and `exit`. Library functions such as printf are built on top of system calls like write.

# Virtual memory and paging

Virtual memory gives each process its own address space, divided into fixed-size pages that the hardware maps to physical frames using a page table. When a process touches a page that is not in memory the hardware raises a page fault and the operating system loads the page, possibly evicting another one chosen by a replacement policy such as FIFO, LRU or the optimal policy. The translation lookaside buffer, or TLB, caches recent page table entries to make translation fast.
