#include "fuzzificator.h"

fuzzificator::fuzzificator()
{
	for (int i = 0; i < 48; i++) {

		fuzzyrules* rule = new fuzzyrules();
		rule->input1 = mFuzzy10BinRulesDefinition[i][0];
		rule->input2 = mFuzzy10BinRulesDefinition[i][1];
		rule->input3 = mFuzzy10BinRulesDefinition[i][2];
		rule->output = mFuzzy10BinRulesDefinition[i][3];
		mFuzzy10BinRules.push_back(rule);
	}

	for (int i = 0; i < 4; i++) {

		fuzzyrules* rule = new fuzzyrules();
		rule->input1 = mFuzzy24BinRulesDefinition[i][0];
		rule->input2 = mFuzzy24BinRulesDefinition[i][1];
		rule->output = mFuzzy24BinRulesDefinition[i][2];
		mFuzzy24BinRules.push_back(rule);
	}

	mHueActivation = std::vector<double>(8);
	mSaturationActivation = std::vector<double>(2);
	mValueActivation = std::vector<double>(3);

	mFuzzy10BinHisto = std::vector<double>(10);
	mFuzzy24BinHisto = std::vector<double>(24);

}

fuzzificator::~fuzzificator()
{
	for (std::vector<fuzzyrules*>::iterator it = mFuzzy10BinRules.begin(); 
		it != mFuzzy10BinRules.end(); ++it)
	{
		delete (*it);
	}
	mFuzzy10BinRules.clear();

	for (std::vector<fuzzyrules*>::iterator it = mFuzzy24BinRules.begin();
		it != mFuzzy24BinRules.end(); ++it)
	{
		delete (*it);
	}
	mFuzzy24BinRules.clear();

	mHueActivation.clear();
	mSaturationActivation.clear();
	mValueActivation.clear();

	mFuzzy10BinHisto.clear();
	mFuzzy24BinHisto.clear();

}

void fuzzificator::calculateFuzzy10BinHisto(
	double hue, double saturation, 
	double value, cv::Mat& fuzzy10BinHisto)
{
	findMembershipValueForTriangles(
		hue, mHueMembershipValues, mHueActivation);

	findMembershipValueForTriangles(
		saturation, 
		mSaturation10BinMembershipValues, 
		mSaturationActivation);

	findMembershipValueForTriangles(
		value, mValue10BinMembershipValues,
		mValueActivation);

	mMultiParticipateDefazzificator(
		mFuzzy10BinRules, mHueActivation, 
		mSaturationActivation, mValueActivation, 
		mFuzzy10BinHisto);


	cv::Mat tmp = cv::Mat(mFuzzy10BinHisto, true).t();
	tmp.copyTo(fuzzy10BinHisto);
}


void fuzzificator::calculateFuzzy24BinHisto(
	double hue, double saturation, double value,
	cv::Mat& colorValues, cv::Mat& fuzzy24BinHisto)
{
	std::vector<double> results = std::vector<double>(3);

	results[0] = 0;
	results[1] = 0;
	results[2] = 0;

	double summColorVals = 0;

	findMembershipValueForTriangles(
		saturation, mSaturation24BinMembershipValues,
		mSaturationActivation);

	findMembershipValueForTriangles(
		value, mValue24BinMembershipValues,
		mValueActivation);

	int cols = colorValues.cols;
	for (int i = 3; i < cols; i++) {
		summColorVals += colorValues.at<double>(i);
	}

	if (summColorVals > 0)
	{
		mMultiParticipateDefazzificator(
			mFuzzy24BinRules, mSaturationActivation,
			mValueActivation, results);
	}


	for (int i = 0; i < 3; i++) {
		mFuzzy24BinHisto[i] += colorValues.at<double>(i);
	}

	for (int i = 3; i < 10; i++) {
		mFuzzy24BinHisto[(i - 2) * 3] += colorValues.at<double>(i) * results[0];
		mFuzzy24BinHisto[(i - 2) * 3 + 1] += colorValues.at<double>(i) * results[1];
		mFuzzy24BinHisto[(i - 2) * 3 + 2] += colorValues.at<double>(i) * results[2];
	}

	cv::Mat tmp = cv::Mat(mFuzzy24BinHisto, true).t();
	tmp.copyTo(fuzzy24BinHisto);
}

void fuzzificator::findMembershipValueForTriangles(
	double input, std::vector<double>& triangles, 
	std::vector<double>& memfunction)
{
	int Temp = 0;

	int size = triangles.size();

	for (int i = 0; i <= size - 1; i += 4) {

		memfunction[Temp] = 0;

		if (input >= triangles[i + 1] && input <= +triangles[i + 2]) 
		{
			memfunction[Temp] = 1;
		}

		if (input >= triangles[i] && input < triangles[i + 1]) 
		{
			memfunction[Temp] = (input - triangles[i]) / (triangles[i + 1] - triangles[i]);
		}

		if (input > triangles[i + 2] && input <= triangles[i + 3]) 
		{
			memfunction[Temp] = (input - triangles[i + 2]) / (triangles[i + 2] - triangles[i + 3]) + 1;
		}

		Temp += 1;
	}
}

void fuzzificator::mMultiParticipateDefazzificator(
	std::vector<fuzzyrules*>& rules, std::vector<double>& input1,
	std::vector<double>& input2, std::vector<double>& input3,
	std::vector<double>& resultTable)
{

	int RuleActivation = -1;

	int size = rules.size();

	for (int i = 0; i < size; i++) {
		if ((input1[rules[i]->input1] > 0) && 
			(input2[rules[i]->input2] > 0) && 
			(input3[rules[i]->input3] > 0)) 
		{

			RuleActivation = rules[i]->output;
			double minimum = 0;
			minimum = std::min(input2[rules[i]->input2], input3[rules[i]->input3]);
			minimum = std::min(input1[rules[i]->input1], minimum);

			resultTable[RuleActivation] += minimum;

		}
	}
}

void fuzzificator::mMultiParticipateDefazzificator(
	std::vector<fuzzyrules*>& rules, std::vector<double>& input1, 
	std::vector<double>& input2, std::vector<double>& resultTable)
{
	int RuleActivation = -1;
	int size = rules.size();
	double minimum = 0;
	for (int i = 0; i < size; i++)
	{
		if ((input1[rules[i]->input1] > 0) && (input2[rules[i]->input2] > 0))
		{
			minimum = std::min(input1[rules[i]->input1], input1[rules[i]->input2]);

			RuleActivation = rules[i]->output;
			resultTable[RuleActivation] += minimum;
		}
	}
}

