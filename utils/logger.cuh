//
// Created by DanielSun on 11/1/2021.
//

#ifndef CUDANNGEN2_LOGGER_CUH
#define CUDANNGEN2_LOGGER_CUH
#include <string>
#include <cstdio>
#include <process.h>
#include <iostream>
#include <Windows.h>
#include <ctime>

using namespace std;

class logger {

};

static string logHeader(){
    time_t rawtime;
    struct tm * timeinfo;
    char buffer[80];

    time (&rawtime);
    timeinfo = localtime(&rawtime);

    strftime(buffer,sizeof(buffer)," %H:%M:%S ",timeinfo);
    std::string str(buffer);

    string out = "[";
    out.append(str).append(" pid:").append(to_string(_getpid())).append(" ] ");
    return out;
}

static void logInfo(const string& info){
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    SetConsoleTextAttribute(hConsole, 0x02);
    printf("%s | I >>> %s \n", logHeader().c_str(), info.c_str());
}

static void logInfo(const char * info){
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    SetConsoleTextAttribute(hConsole, 0x02);
    printf("%s | I >>> %s \n", logHeader().c_str(), info);
}

static void logInfo(const string& info, int color){
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    SetConsoleTextAttribute(hConsole, color);
    printf("%s | I >>> %s \n", logHeader().c_str(), info.c_str());
}

static void logInfo(const char * info, int color){
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    SetConsoleTextAttribute(hConsole, color);
    printf("%s | I >>> %s \n", logHeader().c_str(), info);
}

static void logErr(const string& error){
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    SetConsoleTextAttribute(hConsole, 0x04);
    printf("%s | E >>> %s \n" , logHeader().c_str(), error.c_str());
}

static void logErr(const char * error){
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    SetConsoleTextAttribute(hConsole, 0x04);
    printf("%s | E >>> %s \n" , logHeader().c_str(), error);
}

#endif //CUDANNGEN2_LOGGER_CUH
