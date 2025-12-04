#ifndef FLESCH_H 
#define FLESCH_H

#include <cstdlib>
#include <iomanip>
#include <string>
#include <iostream>

using namespace std;

class FleschReadingEase {
    public:
        FleschReadingEase(string q) { query = q; }
        float get();
    
    private:
        string query;
        static int numSyllables(string);
        static bool isVowel(char);
};

#endif