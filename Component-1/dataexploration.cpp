/*
David Favela Corella
dxf200002
CS 4375.003
*/

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <algorithm>
#include <vector>

double vectorSum(std::vector<double> numericVector) {
    double sumVector = 0;
    for (double eachOne: numericVector){
        sumVector += eachOne;
    }
    return sumVector;
}

double vectorMean(std::vector<double> numericVector) {

    double sumVector = vectorSum(numericVector);
    return sumVector / (numericVector.size());
}

std::vector<double> vectorSorter(std::vector<double> numericVector){
    std::sort(numericVector.begin(), numericVector.end());
    return numericVector;
}

double vectorMedian(std::vector<double> numericVector) {
    numericVector = vectorSorter(numericVector);
    if ((numericVector.size()) % 2 != 0) 
        {
            return numericVector.at((numericVector.size() + 1) /2);
        }
    return (numericVector.at((numericVector.size()-1)/2) + numericVector.at((numericVector.size())/2)) / 2;
}

double vectorRange(std::vector<double> numericVector) {
    numericVector = vectorSorter(numericVector);

    return numericVector.back() - numericVector.front();
}

double vectorCovariance(std::vector<double> numericVector1, std::vector<double> numericVector2) {
    double vectorMean1 = vectorMean(numericVector1);
    double vectorMean2 = vectorMean(numericVector2);
    double adder = 0;

    for(int i = 0; i < numericVector1.size(); i++)
    {
        adder += ((numericVector1.at(i)-vectorMean1) * (numericVector2.at(i)-vectorMean2));
    }

    return (adder / (numericVector1.size() - 1));
}

double vectorDeviation(std::vector<double> numericVector) {
    double vectorMean1 = vectorMean(numericVector);
    double adder = 0;

    for(int i = 0; i < numericVector.size(); i++)
    {
        adder += pow((numericVector.at(i)-vectorMean1),2);
    }

    return sqrt(adder/numericVector.size());
}

double vectorCorrelation(std::vector<double> numericVector1, std::vector<double> numericVector2) {
    
    double vectorCov = vectorCovariance(numericVector1, numericVector2);
    double vectorDev1 = vectorDeviation(numericVector1);
    double vectorDev2 = vectorDeviation(numericVector2);


    return (vectorCov / (vectorDev1 * vectorDev2));
}

void rmStats(std::vector<double> rm){
    std::cout << "\nrm stats" << std::endl;
    std::cout << "Sum: " << vectorSum(rm) << std::endl;
    std::cout << "Mean: " << vectorMean(rm) <<std::endl;
    std::cout << "Median: " << vectorMedian(rm) <<std::endl;
    std::cout << "Range: " << vectorRange(rm) <<std::endl;
}

void medvStats(std::vector<double> mdev){
    std::cout << "\nmdev stats" << std::endl;
    std::cout << "Sum: " << vectorSum(mdev) << std::endl;
    std::cout << "Mean: " << vectorMean(mdev) <<std::endl;
    std::cout << "Median: " << vectorMedian(mdev) <<std::endl;
    std::cout << "Range: " << vectorRange(mdev) <<std::endl;
}


int main(int argc, char **argv)
{
    std::ifstream inFS;
    std::string line;
    std::string rm_in, medv_in;
    const int MAX_LEN = 1000;
    std::vector<double> rm(MAX_LEN);
    std::vector<double> medv(MAX_LEN);

    std::cout << "Opening Boston.csv" << std::endl;

    inFS.open("Boston.csv");
    if (!inFS.is_open())
    {
        std::cout << "Could not open file Boston.csv" << std::endl;
        return 1;
    }
    std::cout << "Reading line 1" << std::endl;
    getline(inFS, line);

    // echo heading
    std::cout << "heading: " << line << std::endl;

    int numObservations = 0;
    while (inFS.good())
    {
        getline(inFS, rm_in, ',');
        getline(inFS, medv_in, '\n');

        rm.at(numObservations) = stof(rm_in);
        medv.at(numObservations) = stof(medv_in);

        numObservations++;
    }

    rm.resize(numObservations);
    medv.resize(numObservations);

    inFS.close(); // closes file

    rmStats(rm);

    medvStats(medv);

    std::cout << "\nCovariance of rm and medv: " << vectorCovariance(rm, medv) <<std::endl;
    std::cout << "Correlation of rm and medv: " << vectorCorrelation(rm, medv) <<std::endl;

    return 0;
}