#ifndef LOG_H
#define LOG_H

#define LOG_D(...) printf("[DEBUG] "); printf(__VA_ARGS__)
#define LOG_T(...) printf("[TRACE] "); printf(__VA_ARGS__)
#define LOG_I(...) printf("[INFO]  "); printf(__VA_ARGS__)
#define LOG_W(...) printf("[WARN]  "); printf(__VA_ARGS__)
#define LOG_E(...) printf("[ERROR] "); printf(__VA_ARGS__)
#define LOG_F(...) printf("[FATAL] "); printf(__VA_ARGS__)

#endif // LOG_H
