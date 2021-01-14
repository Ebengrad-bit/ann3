#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include "FeatureExtraction.h"
#include "MomentsHelper.h"
#include "NV_Recognizer.h"

#include "../FeatureExtractionLib/FeatureExtraction.h"
#include "Visualisation.h"

#include <time.h>
#include <windows.h>
#include <fstream>

#define VAL_FILENAME "value.txt"

using namespace cv;
using namespace std;
using namespace fe;

std::map<int, std::string> val;

MomentsHelper a;
NV_Recognizer rec;

std::shared_ptr<fe::PolynomialManager> polim;
std::shared_ptr<fe::IBlobProcessor> bld;
int imsize = 128;

void generateData()
{
	bool tmp = a.DistributeData("..\\Data\\labeled_data", "..\\Data\\ground_data", "..\\Data\\test_data", 50);
	std::vector<std::string> paths;

	/*tmp = MomentsHelper::GetSamplePaths("Data\\ground_data", paths);
	if (!tmp)
		return;
	if (paths.empty())
		return;*/
	std::map<std::string, std::vector<fe::ComplexMoments>> res_test, res_teach;
		tmp = MomentsHelper::GenerateMoments("..\\Data\\ground_data\\", bld, polim, res_teach);
		if (!tmp)
			return;
		tmp = MomentsHelper::SaveMoments("..\\Data\\teach.txt", res_teach);
		if (!tmp)
			return;
	paths.clear();
	/*tmp = MomentsHelper::GetSamplePaths("Data/test_data", paths);
	if (!tmp)
		return;
	if (paths.empty())
		return;*/
		tmp = MomentsHelper::GenerateMoments("..\\Data\\test_data\\", bld, polim, res_test);
		if (!tmp)
			return;
		tmp = MomentsHelper::SaveMoments("..\\Data\\test.txt", res_test);
	std::cout << "===Gen data!===" << endl;
}

void trainNetwork()
{
	bool tmp;
	std::map<std::string, std::vector<fe::ComplexMoments>> res_teach;
	std::vector<std::string> paths;
	tmp = MomentsHelper::GetSamplePaths("..\\Data\\test_data", paths);
	if (!tmp)
		return;
	if (paths.empty())
		return;
	std::vector<int> l;
	l.push_back(50);
	//l.push_back(50);
	MomentsHelper::ReadMoments("..\\Data\\teach.txt", res_teach);
	tmp = rec.Train(res_teach, l, 10000, 0.01f, 0.1f);
	rec.Save("..\\Data\\ann.txt");
	std::cout << "===Train network!===" << endl;
	if (tmp)
		std::cout << "Trained" << endl;
	else
		std::cout << "Bad Train" << endl;

}

void precisionTest()
{

	rec.Read("..\\Data\\ann.txt");
	std::map<std::string, std::vector<fe::ComplexMoments>> res_test;
	std::vector<std::string> paths;
	bool tmp = MomentsHelper::GetSamplePaths("..\\Data\\ground_data", paths);
	if (!tmp)
		return;
	tmp = MomentsHelper::ReadMoments("..\\Data\\test.txt", res_test);
	std::cout << rec.PrecisionTest(res_test) << " ";
	std::cout << endl;
	std::cout << "===Precision test!===" << endl;
}

void recognizeImage()
{
	rec.Read("..\\Data\\ann.txt");
	string path = ("../Data/numbers2.png");
	Mat image = imread(path, IMREAD_GRAYSCALE);
	imshow("Image to recognize", image);
	//threshold(image, image, 127, 255, CV_THRESH_BINARY);
	vector<Mat> blobs = bld->DetectBlobs(image);
	vector<Mat> nblobs = bld->NormalizeBlobs(blobs, polim->GetBasis()[0][0].first.cols);
	vector<ComplexMoments> res;
	for (int i = 0; i < nblobs.size(); i++)
	{
		res.push_back(polim->Decompose(nblobs[i]));
		string num = rec.Recognize(res.back());
		cout << to_string(i) + " Recognized number: " << num << endl;
		ShowBlobDecomposition("#"+to_string(i) + " Recognized number:" + num, nblobs[i], polim->Recovery(res.back()));
	}
	waitKey(0);
	//destroyAllWindows();
	std::cout << "===Recognize single image!===" << endl;
}

int main(int argc, char** argv)
{
	srand(time(0));
	bld = fe::CreateBlobProcessor();
	polim = fe::CreatePolynomialManager();
	polim->InitBasis(25, imsize);
	string key;
	do 
	{
		std::cout << "===Enter next values to do something:===" << endl;
		std::cout << "  '1' - to generate data." << endl;
		std::cout << "  '2' - to train network." << endl;
		std::cout << "  '3' - to check recognizing precision." << endl;
		std::cout << "  '4' - to recognize single image." << endl;
		std::cout << "  'exit' - to close the application." << endl;
		cin >> key;
		std::cout << endl;
		if (key == "1") {
			generateData();
		}
		else if (key == "2") {
			trainNetwork();
		}
		else if (key == "3") {
			precisionTest();
		}
		else if (key == "4") {
			recognizeImage();
		}
		std::cout << endl;
	} while (key != "exit");
	return 0;
}