# Multithreaded HTTP Server

This project is a multithreaded HTTP server written in C. The server supports basic HTTP methods, ```GET``` and ```PUT```, requests and manages concurrency using a thread pool and a reader-writer locking mechanism for shared resource access.

## Main Components

### 1. Request Handling
The server extracts the method, URI, and other headers from incoming HTTP requests. It validates the request format against predefined rules, returning appropriate HTTP error responses for invalid requests (e.g., ```400 Bad Request```, ```404 Not Found```). Regex patterns are used to efficiently parse and verify request components.

### 2. Concurrency Management

The server uses a configurable thread pool to handle multiple client requests simultaneously. Access to shared resources is synchronized using reader-writer locks, ensuring thread-safe operations and preventing data races. Incoming connections are stored in a queue, from which worker threads pick up requests for processing.

### 3. Request and Response Formats

#### Request Format:

The server can handle ```GET``` and ```PUT``` requests. ```GET``` fetches and returns the content of the requested file if it exists. ```PUT``` writes or updates the content of the specified file.

#### Examples
```
GET /foo.txt HTTP/1.1\r\n\r\n
```
```
PUT /foo.txt HTTP/1.1\r\nContent-Length: 12\r\n\r\nHello world!
```
#### Response Codes:
```200 OK```: Successful request processing.

```201 Created```: File created successfully during a PUT request.

```400 Bad Request```: Malformed request received.

```403 Forbidden```: Access denied due to permissions or directory issues.

```404 Not Found```: Requested file does not exist.

```500 Internal Server Error```: Issues with server-side processing.

## Requirenments
* Computer with ARM or x86 architecture.
* ```gcc``` (GNU Compiler Collection) installed on your system.

## Usage

#### 1. Compile the server
Run ```make``` to compile the source code.

#### 2. Run the server 
Start the server by specifying the number of threads and port number:
```
./httpserver [-t threads] <port> 
```
* Replace ```<port>``` with the desired port number (e.g., 8080).
* Use ```-t threads``` to set the number of worker threads (default: 4).

#### 3. Send HTTP Requests

Use tools like ```curl``` or a web browser to interact with the server:

```GET``` Example:  ```curl http://localhost:<port>/foo.txt```

```PUT``` Example:  ```curl -X PUT --data-binary @file.txt http://localhost:<port>/foo.txt```

#### 5. Stop the Server
Terminate the server using ``` Ctrl+C```.


