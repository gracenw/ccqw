#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdlib.h>
#include <ctime>
#include <chrono>
#include <thread>
#include "flesch/flesch.hpp"

using namespace std;

string send_to_edgebert(string query) {
    this_thread::sleep_for(1s);
    return "";
}

string send_to_calm(string query) {
    this_thread::sleep_for(1s);
    return "";
}

string send_to_cloud(string query) {
    this_thread::sleep_for(1s);
    return "";
}

int main(int argc, char *argv[])
{   
    // argument parsing
    if (argc < 2) {
        cout << "Please submit a query" << endl;
        exit(1);
    }
    cout << "Query: " << argv[1] << endl;
    string query = argv[1];

    // correct grammatical errors
    

    // get reading ease
    FleschReadingEase reading_ease = FleschReadingEase(query);

    // send to appropriate cores
    string result;
    if (reading_ease.get() > 80) {
        result = send_to_edgebert(query);
    }
    else {
        string encryption = send_to_calm(query);
        result = send_to_cloud(encryption);
    }

    cout << "Result: " << result << endl;
 
    return 0;
}