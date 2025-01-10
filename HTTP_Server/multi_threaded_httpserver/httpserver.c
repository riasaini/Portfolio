#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <arpa/inet.h>
#include <stdbool.h>
#include <regex.h>
#include <stdint.h>
#include <sys/types.h>
#include <pthread.h>
#include <semaphore.h>
#include "helper_funcs.h"
#include "queue.h"
#include "rwlock.h"
#define BUFF_SIZE 2048


// Define response messages for different HTTP statuses
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


// Define the structure of a file node in the linked list
typedef struct node {
    char uri[65];
    rwlock_t *lock;
    struct node *next;
} fnode_t;


// Structure representing a read-write lock queue
typedef struct {
    fnode_t *tail;
    fnode_t *head;
} rw_queue_t;


// create a new file node with a given URI
fnode_t *new_node(char *uri) {
    fnode_t *n = (fnode_t *) malloc(sizeof(fnode_t));
    memset(n->uri, '\0', sizeof(n->uri));
    strcpy(n->uri, uri);
    n->lock = rwlock_new(READERS, 1);
    n->next = NULL;
    return n;
}

// create a new read-write lock queue
rw_queue_t *new_rwqueue() {
    rw_queue_t *q = (rw_queue_t *) malloc(sizeof(rw_queue_t));
    q->head = NULL;
    q->tail = NULL;
    return q;
}

// Global variables for the buffer and lock queue, and a mutex for synchronization
queue_t *buffer;
rw_queue_t *rw_lock;
pthread_mutex_t check;

// handle incoming requests
void *requestHandler() {

    while (true) {
        uintptr_t conn;
        queue_pop(buffer, (void **) &conn); // Pop a connection from the queue
        int fd = conn;

        // If the connection is invalid, continue to the next iteration
        if (fd == -1) {
            fprintf(stderr, "couldnt accept request, fd = -1\n");
            continue;
        }

        char BUFF[BUFF_SIZE]; // Buffer to store incoming data
        memset(BUFF, '\0', sizeof(BUFF));

        // Read request data until the end of the header
        read_until(fd, BUFF, BUFF_SIZE, "\r\n\r\n");

        int ret;
        int len;
        int response_length;
        regex_t preg;
        regmatch_t inputMatch[5];

        // Compile a regex to match the incoming request format
        regcomp(&preg, "([a-zA-Z]{0,8}) /([a-zA-Z0-9.-]{1,63}) HTTP/([0-9][.][0-9])\r\n(.*)",
            REG_EXTENDED);
        ret = regexec(&preg, BUFF, 5, inputMatch, 0);

        // If the request doesn't match the expected format, return 400
        if (ret != 0) {
            regfree(&preg);
            response_length = strlen(r400);
            write_n_bytes(fd, r400, response_length);
            close(fd);
            continue;
        }

        // Extract the HTTP method, URI, version, and headers from the request
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

        // Extract the Request-Id header, if present
        char request_id[BUFF_SIZE];
        memset(request_id, '\0', sizeof(request_id));
        regmatch_t getVal[2];
        regcomp(&preg, "Request-Id: ([ -~]{1,128})\r\n", REG_EXTENDED);
        ret = regexec(&preg, header, 2, getVal, 0);
        if (ret != 0) {
            strcpy(request_id, "0");
        } else {
            len = getVal[1].rm_eo - getVal[1].rm_so;
            strncpy(request_id, header + getVal[1].rm_so, len);
        }

        // Extract the Content-Length header, if present
        char val[BUFF_SIZE];
        memset(val, '\0', sizeof(val));
        bool val_exist = false;
        regmatch_t getVal2[2];
        regcomp(&preg, "Content-Length: ([ -~]{1,128})\r\n", REG_EXTENDED);
        ret = regexec(&preg, header, 2, getVal2, 0);
        if (ret == 0) {
            val_exist = true;
            len = getVal2[1].rm_eo - getVal2[1].rm_so;
            strncpy(val, header + getVal2[1].rm_so, len);
        }

        // Extract the body of the request, if present
        char body[BUFF_SIZE];
        memset(body, '\0', sizeof(body));
        regmatch_t getVal3[2];
        regcomp(&preg, "\r\n\r\n(.*)", REG_EXTENDED);
        ret = regexec(&preg, header, 2, getVal3, 0);
        if (ret == 0) {
            len = getVal3[1].rm_eo - getVal3[1].rm_so;
            strncpy(body, header + getVal3[1].rm_so, len);
        }

        regfree(&preg);

        // Check for valid HTTP version
        if (strcmp(version, "1.1") != 0) {
            response_length = strlen(r505);
            write_n_bytes(fd, r505, response_length);
            fprintf(stderr, "%s,%s,%s,%s\n", method, uri, "505", request_id);
            close(fd);
            continue;
        }

        // Check for valid HTTP method (GET or PUT only)
        if ((strcmp(method, "GET") != 0) && (strcmp(method, "PUT") != 0)) {
            response_length = strlen(r501);
            write_n_bytes(fd, r501, response_length);
            fprintf(stderr, "%s,%s,%s,%s\n", method, uri, "501", request_id);
            close(fd);
            continue;
        }

        // Process GET request
        struct stat isfile;
        if (strcmp(method, "GET") == 0) {
            ret = access(uri, F_OK);
            if (ret != 0) {
                response_length = strlen(r404);
                write_n_bytes(fd, r404, response_length);
                fprintf(stderr, "%s,%s,%s,%s\n", method, uri, "404", request_id);
                close(fd);
                continue;
            }
            
            // Check for read permission
            stat(uri, &isfile);
            ret = access(uri, R_OK);
            if (ret != 0 || S_ISDIR(isfile.st_mode)) {
                response_length = strlen(r403);
                write_n_bytes(fd, r403, response_length);
                fprintf(stderr, "%s,%s,%s,%s\n", method, uri, "403", request_id);
                close(fd);
                continue;
            }
            
            // Use a reader lock for file access
            pthread_mutex_lock(&check);
            bool r_exists = false;
            fnode_t *rtemp;
            if (rw_lock->head == NULL) {
                rtemp = new_node(uri);
                rw_lock->head = rtemp;
                rw_lock->tail = rtemp;
                reader_lock(rtemp->lock);
            } else {
                rtemp = rw_lock->head;
                while (rtemp && !r_exists) {
                    if (strcmp(rtemp->uri, uri) == 0) {
                        r_exists = true;
                        reader_lock(rtemp->lock);
                    } else {
                        rtemp = rtemp->next;
                    }
                }
                if (!r_exists) {
                    rtemp = new_node(uri);
                    rw_lock->tail->next = rtemp;
                    rw_lock->tail = rtemp;
                    reader_lock(rw_lock->tail->lock);
                }
            }
            pthread_mutex_unlock(&check);

            // Open file and send its contents in the response
            int fd2 = open(uri, O_RDONLY);
            if (fd2 < 0) {
                response_length = strlen(r500);
                write_n_bytes(fd, r500, response_length);
                fprintf(stderr, "%s,%s,%s,%s\n", method, uri, "500", request_id);
                
                // Release the reader lock
                fnode_t *t3 = rw_lock->head;
                while (t3) {
                    if (strcmp(t3->uri, uri) == 0) {
                        reader_unlock(t3->lock);
                    }
                    t3 = t3->next;
                }

                // Move to next request
                close(fd);
                continue;
            }

            int curr = lseek(fd2, 0, SEEK_CUR);
            ret = lseek(fd2, 0, SEEK_END);
            lseek(fd2, curr, SEEK_SET);

            // Prepare HTTP response header with content length
            char response[BUFF_SIZE];
            memset(response, '\0', sizeof(response));
            sprintf(response, "HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n", ret);
            write_n_bytes(fd, response, strlen(response));
            pass_n_bytes(fd2, fd, ret);
            fprintf(stderr, "%s,%s,%s,%s\n", method, uri, "200", request_id);
            close(fd2);

            // Release the reader lock
            fnode_t *t2 = rw_lock->head;
            while (t2) {
                if (strcmp(t2->uri, uri) == 0) {
                    reader_unlock(t2->lock);
                }
                t2 = t2->next;
            }
            // Move to next request
            close(fd);
            continue;
        }
        // Process PUT request
        if (strcmp(method, "PUT") == 0) {
            // Check if the value exists for PUT request
            if (!val_exist) {
                response_length = strlen(r400);
                write_n_bytes(fd, r400, response_length);
                fprintf(stderr, "%s,%s,%s,%s\n", method, uri, "400", request_id);
                close(fd);
                continue;
            }

            // Use a writer lock for file access
            pthread_mutex_lock(&check);
            bool wexist = false;
            fnode_t *temp;
            if (rw_lock->head == NULL) {
                temp = new_node(uri);
                rw_lock->head = temp;
                rw_lock->tail = temp;
                writer_lock(temp->lock);
            } else {
                temp = rw_lock->head;
                while (temp && !wexist) {
                    if (strcmp(temp->uri, uri) == 0) {
                        wexist = true;
                        writer_lock(temp->lock);
                    } else {
                        temp = temp->next;
                    }
                }
                if (!wexist) {
                    temp = new_node(uri);
                    rw_lock->tail->next = temp;
                    rw_lock->tail = temp;
                    writer_lock(rw_lock->tail->lock);
                }
            }
            pthread_mutex_unlock(&check);

            // Check if the file exists and if write permission is available
            ret = access(uri, F_OK);
            if (ret == 0) {
                stat(uri, &isfile);
                ret = access(uri, W_OK);
                if (ret != 0 || S_ISDIR(isfile.st_mode)) {
                    response_length = strlen(r403);
                    write_n_bytes(fd, r403, response_length);
                    fprintf(stderr, "%s,%s,%s,%s\n", method, uri, "403", request_id);
                    close(fd);

                    // Release the writer lock
                    fnode_t *temp3 = rw_lock->head;
                    while (temp3) {
                        if (strcmp(temp3->uri, uri) == 0) {
                            writer_unlock(temp3->lock);
                        }
                        temp3 = temp3->next;
                    }
                    continue;
                }
            }

            // Open file for writing (create or overwrite)
            int fd3 = open(uri, O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (fd3 < 0) {
                response_length = strlen(r500);
                write_n_bytes(fd, r500, response_length);
                fprintf(stderr, "%s,%s,%s,%s\n", method, uri, "500", request_id);
                close(fd);
                // Release the writer lock
                fnode_t *temp4 = rw_lock->head;
                while (temp4) {
                    if (strcmp(temp4->uri, uri) == 0) {
                        writer_unlock(temp4->lock);
                    }
                    temp4 = temp4->next;
                }

                continue;
            }
            // Write data to the file
            ret = write_n_bytes(fd3, body, strlen(body));
            if (ret < atoi(val)) {
                ret = pass_n_bytes(fd, fd3, (atoi(val) - ret));
            }
            close(fd3);

            // Check if writing was successful and send appropriate response
            if (ret < 0) {
                response_length = strlen(r500);
                write_n_bytes(fd, r500, response_length);
                fprintf(stderr, "%s,%s,%s,%s\n", method, uri, "500", request_id);
                close(fd);
            } else if (!wexist) {
                response_length = strlen(r201);
                write_n_bytes(fd, r201, response_length);
                fprintf(stderr, "%s,%s,%s,%s\n", method, uri, "201", request_id);
                close(fd);
            } else {
                response_length = strlen(r200);
                write_n_bytes(fd, r200, response_length);
                fprintf(stderr, "%s,%s,%s,%s\n", method, uri, "200", request_id);
                close(fd);
            }

            // Release the writer lock
            fnode_t *temp2 = rw_lock->head;
            while (temp2) {
                if (strcmp(temp2->uri, uri) == 0) {
                    writer_unlock(temp2->lock);
                }
                temp2 = temp2->next;
            }

            continue;
        }
    }
}

// Main server function that initializes listener socket and handles multiple threads
int main(int argc, char *argv[]) {
    int num_threads = 4;
    int port;
    int opt;
    // Parse command-line arguments for thread count and port number
    while ((opt = getopt(argc, argv, "t:")) != -1) {
        switch (opt) {
        case 't': num_threads = atoi(optarg); break;
        default: break;
        }
    }
    if (optind < argc) {
        port = atoi(argv[optind]);
    } else {
        fprintf(stderr, "USAGE: ./httpserver [-t threads] <port>\n");
        return 1; // Exit if no valid port is provided
    }
    // Initialize listener socket and handle errors
    Listener_Socket listener_socket;
    int ret = listener_init(&listener_socket, port);
    if (ret == -1) {
        fprintf(stderr, "%s", "Invalid Port\n");
        exit(1);
    }

    // Initialize synchronization primitives and buffers
    pthread_mutex_init(&check, NULL);
    buffer = queue_new(num_threads);
    rw_lock = new_rwqueue();

    // Create worker threads
    pthread_t *worker_threads = (pthread_t *) malloc(sizeof(pthread_t) * num_threads);
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&worker_threads[i], NULL, &requestHandler, NULL);
    }

    // Accept client connections and push them to buffer
    while (true) {
        uintptr_t connection_fd = listener_accept(&listener_socket);
        queue_push(buffer, (void *) connection_fd);
    }
}
