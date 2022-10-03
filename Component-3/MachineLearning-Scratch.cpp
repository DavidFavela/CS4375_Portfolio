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
#include <chrono>

// accuracy , sensitivity, specificity

double age_lhFunction(double v, double mean, double var);

double accuracyFunc(double accurateOnes, double testDataSize)
{
    return accurateOnes / testDataSize;
}

double sensitivityFunc(double truePositives, double tPfN)
{
    return truePositives / tPfN;
}

double specificityFunc(double trueNegatives, double tNfP)
{
    return trueNegatives / tNfP;
}

std::vector<double> raw_probFunction(double pClass, double sex, double age, std::vector<std::vector<double>> lh_pclass, std::vector<std::vector<double>> lh_sex,
                                     double aPrioriSurv, double aPrioriNotSurv, std::vector<std::vector<double>> ageMean, std::vector<std::vector<double>> ageVar)
{
    std::vector<double> probs = {0, 0};

    double num_s = lh_pclass[pClass - 1][1] * lh_sex[1][sex] * aPrioriSurv * age_lhFunction(age, ageMean[0][1], ageVar[0][1]);
    double num_p = lh_pclass[pClass - 1][0] * lh_sex[0][sex] * aPrioriNotSurv * age_lhFunction(age, ageMean[0][0], ageVar[0][0]);
    double denominator = lh_pclass[pClass - 1][1] * lh_sex[1][sex] * aPrioriSurv * age_lhFunction(age, ageMean[0][1], ageVar[0][1]) + lh_pclass[pClass - 1][0] * lh_sex[0][sex] * aPrioriNotSurv * age_lhFunction(age, ageMean[0][0], ageVar[0][0]);

    probs[1] = num_s / denominator;
    probs[0] = num_p / denominator;

    return probs;
}

double age_lhFunction(double v, double mean, double var)
{
    double age_lh = 1 / ((sqrt(2 * M_PI * var)) * exp((-(((v - mean) * (v - mean)) / (2 * var)))));
    return age_lh;
}

double getVariance(std::vector<double> dataSet, double mean)
{
    double sumofData = 0;

    for (int i = 0; i < dataSet.size(); i++)
    {
        double tothepower = pow((dataSet[i] - mean), 2);
        sumofData = tothepower + sumofData;
    }
    sumofData = sumofData / dataSet.size();

    return sumofData;
}

std::vector<std::vector<double>> predict(std::vector<std::vector<double>> probabilities)
{
    std::vector<double> row2(probabilities.at(0).size(), 0);
    std::vector<std::vector<double>> z(1, row2);

    for (int i = 0; i < z.at(0).size(); i++)
    {
        if (probabilities[0][i] > 0.5)
            z[0][i] = 1;
        else
            z[0][i] = 0;
    }
    return z;
}

std::vector<std::vector<double>> probFunc(std::vector<std::vector<double>> predicted)
{
    std::vector<double> row2(predicted.at(0).size(), 0);
    std::vector<std::vector<double>> z(1, row2);

    for (int i = 0; i < z.at(0).size(); i++)
    {
        z[0][i] = (exp(predicted[0][i])) / (1 + exp(predicted[0][i]));
    }
    return z;
}

std::vector<std::vector<double>> matrixMult(std::vector<std::vector<double>> dataFrame, std::vector<std::vector<double>> weights)
{
    // Initializing a single row
    std::vector<double> row2(dataFrame.at(0).size(), 0);
    // Initializing the 2-D vector
    std::vector<std::vector<double>> z(1, row2);

    for (int i = 0; i < z.at(0).size(); i++)
    {
        z[0][i] = dataFrame[0][i] * weights[0][0] + dataFrame[1][i] * weights[0][1];
    }

    return z;
}

std::vector<std::vector<double>> weightFunction(std::vector<std::vector<double>> weights, double learningRate, std::vector<std::vector<double>> dataMatrix, std::vector<std::vector<double>> error)
{

    std::vector<std::vector<double>> carry{{0, 0}};

    for (int i = 0; i < dataMatrix.at(0).size(); i++)
    {
        carry[0][0] = dataMatrix[0][i] * error[0][i] + carry[0][0];
        // std::cout << i << " " << dataMatrix[0][i] << " " << error[0][i];
        carry[0][1] = dataMatrix[1][i] * error[0][i] + carry[0][1];
        // std::cout <<" " << dataMatrix[1][i] << " " << error[0][i] << std::endl;
    }

    carry[0][0] = carry[0][0] * learningRate;
    carry[0][1] = carry[0][1] * learningRate;

    weights[0][0] = weights[0][0] + carry[0][0];
    weights[0][1] = weights[0][1] + carry[0][1];

    return weights;
}

std::vector<std::vector<double>> errorFunction(std::vector<std::vector<double>> labels, std::vector<std::vector<double>> prob_vector)
{
    std::vector<double> row(labels.at(0).size(), 0);
    std::vector<std::vector<double>> error(1, row);

    for (int i = 0; i < error.at(0).size(); i++)
    {
        error[0][i] = labels[0][i] - prob_vector[0][i];
    }

    return error;
}

std::vector<std::vector<double>> sigmoid(std::vector<std::vector<double>> dataFrame, std::vector<std::vector<double>> weights)
{
    // Initializing a single row
    std::vector<double> row2(dataFrame.at(0).size(), 0);
    // Initializing the 2-D vector
    std::vector<std::vector<double>> z(1, row2);

    for (int i = 0; i < z.at(0).size(); i++)
    {
        double eachZ = -(dataFrame[0][i] * weights[0][0] + dataFrame[1][i] * weights[0][1]);
        z[0][i] = 1 / (1 + exp(eachZ));
    }

    return z;
}

/*
double vectorSum(std::vector<double> numericVector)
{
    double sumVector = 0;
    for (double eachOne : numericVector)
    {
        sumVector += eachOne;
    }
    return sumVector;
}

double vectorMean(std::vector<double> numericVector)
{

    double sumVector = vectorSum(numericVector);
    return sumVector / (numericVector.size());
}

std::vector<double> vectorSorter(std::vector<double> numericVector)
{
    std::sort(numericVector.begin(), numericVector.end());
    return numericVector;
}

double vectorMedian(std::vector<double> numericVector)
{
    numericVector = vectorSorter(numericVector);
    if ((numericVector.size()) % 2 != 0)
    {
        return numericVector.at((numericVector.size() + 1) / 2);
    }
    return (numericVector.at((numericVector.size() - 1) / 2) + numericVector.at((numericVector.size()) / 2)) / 2;
}

double vectorRange(std::vector<double> numericVector)
{
    numericVector = vectorSorter(numericVector);

    return numericVector.back() - numericVector.front();
}

double vectorCovariance(std::vector<double> numericVector1, std::vector<double> numericVector2)
{
    double vectorMean1 = vectorMean(numericVector1);
    double vectorMean2 = vectorMean(numericVector2);
    double adder = 0;

    for (int i = 0; i < numericVector1.size(); i++)
    {
        adder += ((numericVector1.at(i) - vectorMean1) * (numericVector2.at(i) - vectorMean2));
    }

    return (adder / (numericVector1.size() - 1));
}

double vectorDeviation(std::vector<double> numericVector)
{
    double vectorMean1 = vectorMean(numericVector);
    double adder = 0;

    for (int i = 0; i < numericVector.size(); i++)
    {
        adder += pow((numericVector.at(i) - vectorMean1), 2);
    }

    return sqrt(adder / numericVector.size());
}

double vectorCorrelation(std::vector<double> numericVector1, std::vector<double> numericVector2)
{

    double vectorCov = vectorCovariance(numericVector1, numericVector2);
    double vectorDev1 = vectorDeviation(numericVector1);
    double vectorDev2 = vectorDeviation(numericVector2);

    return (vectorCov / (vectorDev1 * vectorDev2));
}

void rmStats(std::vector<double> survived)
{
    std::cout << "\nrm stats" << std::endl;
    std::cout << "Sum: " << vectorSum(survived) << std::endl;
    std::cout << "Mean: " << vectorMean(survived) << std::endl;
    std::cout << "Median: " << vectorMedian(survived) << std::endl;
    std::cout << "Range: " << vectorRange(survived) << std::endl;
}

void medvStats(std::vector<double> mdev)
{
    std::cout << "\nmdev stats" << std::endl;
    std::cout << "Sum: " << vectorSum(mdev) << std::endl;
    std::cout << "Mean: " << vectorMean(mdev) << std::endl;
    std::cout << "Median: " << vectorMedian(mdev) << std::endl;
    std::cout << "Range: " << vectorRange(mdev) << std::endl;
}
*/

int main(int argc, char **argv)
{
    std::ifstream inFS;
    std::string line;
    std::string survived_in, sex_in, pclass_in, age_in;
    const int MAX_LEN = 1047;
    std::vector<double> survived(MAX_LEN);
    std::vector<double> sex(MAX_LEN);
    std::vector<double> pclass(MAX_LEN);
    std::vector<double> age(MAX_LEN);
    std::vector<double> trainSurvived(800);
    std::vector<double> trainSex(800);
    std::vector<double> testSurvived(MAX_LEN - 800);
    std::vector<double> testSex(MAX_LEN - 800);

    std::cout << "Opening titanic_project.csv" << std::endl;

    inFS.open("titanic_project.csv");
    if (!inFS.is_open())
    {
        std::cout << "Could not open file titanic_project.csv" << std::endl;
        return 1;
    }
    getline(inFS, line);

    int numObservations = 0;
    while (inFS.good())
    {
        std::string nomatter1, nomatter2;
        getline(inFS, nomatter1, ',');
        getline(inFS, pclass_in, ',');
        getline(inFS, survived_in, ',');
        getline(inFS, sex_in, ',');
        getline(inFS, age_in, '\n');

        pclass.at(numObservations) = stof(pclass_in);
        survived.at(numObservations) = stof(survived_in);
        sex.at(numObservations) = stof(sex_in);
        age.at(numObservations) = stof(age_in);

        numObservations++;
    }

    pclass.resize(numObservations);
    survived.resize(numObservations);
    sex.resize(numObservations);
    age.resize(numObservations);

    inFS.close(); // closes file

    // Initializing a single row
    std::vector<double> row(numObservations, 0);
    std::vector<double> row800(800, 0);
    std::vector<double> rowrest(numObservations - 801, 0);

    // Initializing the 2-D vector
    std::vector<std::vector<double>> ogData(4, row);
    std::vector<std::vector<double>> trainData(4, row800);
    std::vector<std::vector<double>> testData(4, rowrest);

    for (int i = 0; i < survived.size() - 1; i++)
    {
        ogData[0][i] = survived.at(i);
        ogData[1][i] = sex.at(i);
        ogData[2][i] = pclass.at(i);
        ogData[3][i] = age.at(i);
        if (i < 800)
        {
            trainData[0][i] = survived.at(i);
            trainData[1][i] = sex.at(i);
            trainData[2][i] = pclass.at(i);
            trainData[3][i] = age.at(i);
        }
        else
        {
            testData[0][i - 800] = (survived.at(i));
            testData[1][i - 800] = (sex.at(i));
            testData[2][i - 800] = (pclass.at(i));
            testData[3][i - 800] = (age.at(i));
        }
    }
    // std::cout << testData.at(1).size() << std::endl;

    std::vector<std::vector<double>> weight{{1, 1}};

    std::vector<double> row2(2, 0);
    std::vector<double> row3(trainData.at(0).size(), 0);
    std::vector<std::vector<double>> labels(1, row3);
    std::vector<std::vector<double>> dataMatrix = trainData;
    std::vector<std::vector<double>> dataMatrixTest = testData;
    std::vector<std::vector<double>> error(1, row3);

    for (int i = 0; i < dataMatrix.at(0).size(); i++)
    {
        dataMatrix[0][i] = 1;
    }

    for (int i = 0; i < testData.at(0).size(); i++)
    {
        dataMatrixTest[0][1] = 1;
    }

    labels[0] = trainData[0];

    double learningRate = 0.001;
    std::vector<std::vector<double>> prob_vector(1, row2);

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < 500000; i++)
    {
        prob_vector = sigmoid(dataMatrix, weight);
        error = errorFunction(labels, prob_vector);
        weight = weightFunction(weight, learningRate, dataMatrix, error);
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;

    // PREDICTING STUFF

    std::vector<std::vector<double>> predicted = matrixMult(dataMatrixTest, weight);
    std::vector<std::vector<double>> probabilities = probFunc(predicted);
    std::vector<std::vector<double>> prediction = predict(probabilities);

    std::cout << "Predicted From Test Matrix: " << std::endl;

    for (int i = 0; i < predicted.at(0).size(); i++)
    {
        std::cout << predicted[0][i] << " | ";
    }

    std::cout << "\n\n Probabilities for Test Matrix: " << std::endl;

    for (int i = 0; i < probabilities.at(0).size(); i++)
    {
        std::cout << probabilities[0][i] << " | ";
    }

    std::cout << "\n\n Prediction for Test Matrix: " << std::endl;

    for (int i = 0; i < prediction.at(0).size(); i++)
    {
        std::cout << prediction[0][i] << " | ";
    }

    // Accuracy, sensitivity, specificity for Logistic

    int accurateOnes = 0;
    int truePositives = 0;
    int trueNegatives = 0;
    int falsePositives = 0;
    int falseNegatives = 0;
    for (int i = 0; i < testData.at(0).size(); i++)
    {
        if (prediction[0][i] == testData[0][i] && prediction[0][i] == 1)
            truePositives++;
        else if (prediction[0][i] == testData[0][i] && prediction[0][i] == 0)
            trueNegatives++;
        else if (prediction[0][i] != testData[0][i] && prediction[0][i] == 1 && testData[0][i] == 0)
            falsePositives++;
        else if (prediction[0][i] != testData[0][i] && prediction[0][i] == 0 && testData[0][i] == 1)
            falseNegatives++;
    }

    std::cout << "\n\nElapsed time of the training algorithm: " << elapsed_seconds.count() << "s";

    double accuracy = accuracyFunc(trueNegatives + truePositives, testData.at(0).size());
    std::cout << "\n\nAccuracy: " << accuracy << std::endl;
    double sensitivity = sensitivityFunc(truePositives, truePositives + falseNegatives);
    std::cout << "\nSensitivity: " << sensitivity << std::endl;
    double specificity = specificityFunc(trueNegatives, trueNegatives + falsePositives);
    std::cout << "\nSpecificity: " << specificity << std::endl;

    // NAIVE BAYES

    // A priori values
    double aPrioriSurv = 0;
    for (int i = 0; i < trainData.at(0).size(); i++)
    {

        aPrioriSurv = trainData[0][i] + aPrioriSurv;
    }

    // Num of Survived and not survived
    double survivedNum = aPrioriSurv;
    double notSurvivedNum = (trainData.at(0).size()) - survivedNum;

    // Prob of surv or not surv
    aPrioriSurv = aPrioriSurv / (trainData.at(0).size());
    double aPrioriNotSurv = 1 - aPrioriSurv;

    // Likelihood for qualitative data
    // For PClass
    // Need to do following: pclass = 1 for people that died divided by all people that died and so on

    std::vector<std::vector<double>> lh_pclass{{0, 0}, {0, 0}, {0, 0}};

    for (int i = 0; i < trainData.at(0).size(); i++)
    {
        // pclass 1
        if (trainData[2][i] == 1)
        {
            if (trainData[0][i] == 0)
                lh_pclass[0][0] = 1 + lh_pclass[0][0];
            else
                lh_pclass[0][1] = 1 + lh_pclass[0][1];
        }
        // pclass 2
        if (trainData[2][i] == 2)
        {
            if (trainData[0][i] == 0)
                lh_pclass[1][0] = 1 + lh_pclass[1][0];
            else
                lh_pclass[1][1] = 1 + lh_pclass[1][1];
        }
        // pclass 3
        if (trainData[2][i] == 3)
        {
            if (trainData[0][i] == 0)
                lh_pclass[2][0] = 1 + lh_pclass[2][0];
            else
                lh_pclass[2][1] = 1 + lh_pclass[2][1];
        }
    }

    for (int i = 0; i < lh_pclass.size(); i++)
    {
        // not survived
        lh_pclass[i][0] = lh_pclass[i][0] / notSurvivedNum;
        // survived
        lh_pclass[i][1] = lh_pclass[i][1] / survivedNum;
    }

    // For Sex
    std::vector<std::vector<double>> lh_sex{{0, 0}, {0, 0}};

    for (int i = 0; i < trainData.at(0).size(); i++)
    {
        if (trainData[1][i] == 0) // not survived
        {
            if (trainData[0][i] == 0) // men
                lh_sex[0][0] = 1 + lh_sex[0][0];
            else // women
                lh_sex[0][1] = 1 + lh_sex[0][1];
        }
        if (trainData[1][i] == 1) // survived
        {
            if (trainData[0][i] == 0) // men
                lh_sex[1][0] = 1 + lh_sex[1][0];
            else // women
                lh_sex[1][1] = 1 + lh_sex[1][1];
        }
    }

    for (int i = 0; i < lh_sex.size(); i++)
    {
        lh_sex[i][0] = lh_sex[i][0] / notSurvivedNum;
        lh_sex[i][1] = lh_sex[i][1] / survivedNum;
    }

    // Likelihood for quantitative Data
    std::vector<std::vector<double>> age_mean{{0, 0}};
    std::vector<std::vector<double>> age_variance{{0, 0}};
    std::vector<double> vectorDeath;
    std::vector<double> vectorSurvive;

    double sumofAlive = 0;
    double sumofDeath = 0;
    for (int i = 0; i < trainData.at(0).size(); i++)
    {
        // if death
        if (trainData[0][i] == 0)
        {
            sumofDeath = trainData[3][i] + sumofDeath;
            vectorDeath.push_back(trainData[3][i]);
        }
        else // if survived
        {
            sumofAlive = trainData[3][i] + sumofAlive;
            vectorSurvive.push_back(trainData[3][i]);
        }
    }

    age_mean[0][0] = sumofDeath / notSurvivedNum; // mean for not survived
    age_mean[0][1] = sumofAlive / survivedNum;    // mean for survived

    // get the variance

    age_variance[0][0] = getVariance(vectorDeath, age_mean[0][0]);   // variance for not survived
    age_variance[0][1] = getVariance(vectorSurvive, age_mean[0][1]); // variance for survived

    // RAW PROB

    std::cout << "\nProbabilities in Naive Bayes: " << std::endl;

    for (int i = 0; i < testData.at(0).size(); i++)
    {
        std::vector<double> tryProb = raw_probFunction(testData[2][i], testData[1][i], testData[3][i], lh_pclass, lh_sex, aPrioriSurv, aPrioriNotSurv, age_mean, age_variance);
        std::cout << tryProb[0] << " " << tryProb[1] << " | ";
    }

    // std::cout << tryProb[0] << " " << tryProb[1] << std::endl;

    return 0;
}