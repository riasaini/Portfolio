#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <sys/stat.h>
#include <stdbool.h>
#include <regex.h>
#include "helper_funcs.h"

// HTTP response strings for different status codes
char r200[] = "HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\nOK\n";
char r201[] = "HTTP/1.1 201 Created\r\nContent-Length: 8\r\n\r\nCreated\n";
char r400[] = "HTTP/1.1 400 Bad Request\r\nContent-Length: 12\r\n\r\nBad Request\n";
char r403[] = "HTTP/1.1 403 Forbidden\r\nContent-Length: 10\r\n\r\nForbidden\n";
char r404[] = "HTTP/1.1 404 Not Found\r\nContent-Length: 10\r\n\r\nNot Found\n";
char r500[]
    = "HTTP/1.1 500 Internal Server Error\r\nContent-Length: 21\r\n\r\nInternal Server Error\n";
char r501[] = "HTTP/1.1 501 Not Implemented\r\nContent-Length: 16\r\n\r\nNot Implemented\n";
char r505[]
    = "HTTP/1.1 505 Version Not Supported\r\nContent-Length: 22\r\n\r\nVersion Not Supported\n";

int main(int argc, char *argv[]) {
    // Validate command-line arguments (expects port number as argument)
    if (argc != 2) {
        return 1;
    }

    // Parse and validate port number
    int port = atoi(argv[1]);
    if (port < 1 || port > 65535) {
        fprintf(stderr, "%s", "Invalid Port\n");
        return 1;
    }

    // Initialize listener socket for incoming connections
    Listener_Socket listener_socket;
    listener_socket.fd = 0;
    int ret = listener_init(&listener_socket, port);
    if (ret == -1) {
        fprintf(stderr, "%s", "Invalid Port coudlnt connect\n");
        return 1;
    }

    int len;
    int response_length;
    int BUFF_SIZE = 2048;
    while (true) {
        char BUFF[BUFF_SIZE];
        memset(BUFF, '\0', sizeof(BUFF));

        // Accept incoming connections
        int fd = listener_accept(&listener_socket);
        if (fd == -1) {
            fprintf(stderr, "couldnt accept request, fd = -1\n");
            continue;
        }

        // Read HTTP request until the end of headers (\r\n\r\n)
        read_until(fd, BUFF, BUFF_SIZE, "\r\n\r\n");

        // Parse HTTP request using regex
        regex_t preg;
        regmatch_t inputMatch[5];
        regcomp(&preg, "([a-zA-Z]{0,8}) /([a-zA-Z0-9.-]{1,63}) HTTP/([0-9][.][0-9])\r\n(.*)",
            REG_EXTENDED);
        ret = regexec(&preg, BUFF, 5, inputMatch, 0);
        if (ret != 0) {
            regfree(&preg);
            response_length = strlen(r400);
            write_n_bytes(fd, r400, response_length);
            close(fd);
            continue;
        }

        // Extract method, URI, version, and headers from the request
        len = inputMatch[1].rm_eo - inputMatch[1].rm_so;
        char method[len + 1];
        memset(method, '\0', sizeof(method));
        strncpy(method, BUFF + inputMatch[1].rm_so, len);

        len = inputMatch[2].rm_eo - inputMatch[2].rm_so;
        char uri[len + 1];
        memset(uri, '\0', sizeof(uri));
        strncpy(uri, BUFF + inputMatch[2].rm_so, len);

        len = inputMatch[3].rm_eo - inputMatch[3].rm_so;
        char version[len + 1];
        memset(version, '\0', sizeof(version));
        strncpy(version, BUFF + inputMatch[3].rm_so, len);

        len = inputMatch[4].rm_eo - inputMatch[4].rm_so;
        char header[len + 1];
        memset(header, '\0', sizeof(header));
        strncpy(header, BUFF + inputMatch[4].rm_so, len);

        // Check for supported HTTP version
        if (strcmp(version, "1.1") != 0) {
            regfree(&preg);
            response_length = strlen(r505);
            write_n_bytes(fd, r505, response_length);
            close(fd);
            continue;
        }

        // Check for supported HTTP methods (GET and PUT only)
        if ((strcmp(method, "GET") != 0) && (strcmp(method, "PUT") != 0)) {
            regfree(&preg);
            response_length = strlen(r501);
            write_n_bytes(fd, r501, response_length);
            close(fd);
            continue;
        }

        // Check for Content-Length header if required
        bool valExist = false;
        char val[BUFF_SIZE];
        memset(val, '\0', sizeof(val));

        regmatch_t getVal[2];
        regcomp(&preg, "Content-Length: ([ -~]{1,128})", REG_EXTENDED);
        ret = regexec(&preg, header, 2, getVal, 0);
        if (ret == 0) {
            valExist = true;
            len = getVal[1].rm_eo - getVal[1].rm_so;
            strncpy(val, header + getVal[1].rm_so, len);
        }

        // Extract request body if present
        char body[BUFF_SIZE];
        memset(body, '\0', sizeof(body));

        regmatch_t getBody[2];
        regcomp(&preg, "\r\n\r\n(.*)", REG_EXTENDED);
        ret = regexec(&preg, header, 2, getBody, 0);
        if (ret == 0) {
            len = getBody[1].rm_eo - getBody[1].rm_so;
            strncpy(body, header + getBody[1].rm_so, len);
        }

        regfree(&preg);

        struct stat isfile;
        // Handle GET method
        if (strcmp(method, "GET") == 0) {
            
            // Check if file exists
            ret = access(uri, F_OK);
            if (ret != 0) {
                response_length = strlen(r404);
                write_n_bytes(fd, r404, response_length);
                close(fd);
                continue;
            }

            // Check file permissions
            stat(uri, &isfile);
            ret = access(uri, R_OK);
            if (ret != 0 || S_ISDIR(isfile.st_mode)) {
                response_length = strlen(r403);
                write_n_bytes(fd, r403, response_length);
                close(fd);
                continue;
            }

            // Open and read the file
            int fd2 = open(uri, O_RDONLY);
            if (fd2 < 0) {
                response_length = strlen(r500);
                write_n_bytes(fd, r500, response_length);
                close(fd);
                continue;
            }

            // Calculate file size and send response
            int curr = lseek(fd2, 0, SEEK_CUR);
            ret = lseek(fd2, 0, SEEK_END);
            lseek(fd2, curr, SEEK_SET);

            char response[BUFF_SIZE];
            memset(response, '\0', sizeof(response));
            sprintf(response, "HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n", ret);
            write_n_bytes(fd, response, strlen(response));
            pass_n_bytes(fd2, fd, ret);
            close(fd2);
            close(fd);
            continue;
        }

        // Handle PUT method
        if (strcmp(method, "PUT") == 0) {
            
            // Validate Content-Length header
            if (!valExist) {
                response_length = strlen(r400);
                write_n_bytes(fd, r400, response_length);
                close(fd);
                continue;
            }

            // Check write permission on the file
            // If the file is not writable or is a directory, send HTTP 403 (Forbidden)
            bool fileExistance = false;
            ret = access(uri, F_OK);
            if (ret == 0) {
                fileExistance = true;
                stat(uri, &isfile);
                ret = access(uri, W_OK);
                if (ret != 0 || S_ISDIR(isfile.st_mode)) {
                    response_length = strlen(r403);
                    write_n_bytes(fd, r403, response_length);
                    close(fd);
                    continue;
                }
            }

            // Open the file for writing (create it if it doesn't exist, truncate if it does)
            int fd3 = open(uri, O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (fd3 < 0) {
                response_length = strlen(r500);
                write_n_bytes(fd, r500, response_length);
                close(fd);
                continue;
            }

            // Write the body data to the file
            ret = write_n_bytes(fd3, body, strlen(body));
            if (ret < atoi(val)) {
                int ret2 = pass_n_bytes(fd, fd3, (atoi(val) - ret));
                if (ret2 < 0) {
                    response_length = strlen(r500);
                    write_n_bytes(fd, r500, response_length);
                    close(fd3);
                    close(fd);
                    continue;
                }
            }
            close(fd3);

            if (!fileExistance) {
                response_length = strlen(r201);
                ret = write_n_bytes(fd, r201, response_length);
            } else {
                response_length = strlen(r200);
                ret = write_n_bytes(fd, r200, response_length);
            }
            // Close the connection
            close(fd);
            continue;
        }
    }
}
