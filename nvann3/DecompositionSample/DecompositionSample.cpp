#include <iostream>
#include <FeatureExtraction.h>

#include "Visualisation.h"

using namespace cv;
using namespace fe;
using namespace std;

int main() 
{
	cout << "hello world!" << endl;
	cout << fe::GetTestString().c_str() << endl;

	int max;
	int diametr;

	cout << "Max polinom:";
	cin >> max;
	cout << endl;
	cout << "Diametr:";
	cin >> diametr;
	cout << endl;

	auto PoliMan = fe::CreatePolynomialManager();
	auto BlobPoc = fe::CreateBlobProcessor();

	PoliMan->InitBasis(max, diametr); 
	ShowPolynomials("Basic:", PoliMan->GetBasis()); 

	cv::Mat image = cv::imread("..\\DecompositionSample\\Picture.png", CV_8UC1); 
	imshow("Picture:", image); 

	cout << PoliMan->GetType() << endl << endl;

	vector<cv::Mat>blobs;
	//vector<cv::Mat> normalized_blobs;

	blobs = BlobPoc->DetectBlobs(image); 
	blobs = BlobPoc->NormalizeBlobs(blobs, diametr); 

	cout << BlobPoc->GetType() << endl;

	vector<fe::ComplexMoments> blobs_decompos; 
	blobs_decompos.resize(blobs.size()); 
	vector<cv::Mat> Recovery_blobs;
	Recovery_blobs.resize(blobs.size()); 

	for (int i = 0; i < Recovery_blobs.size(); i++)
	{
		Recovery_blobs[i] = cv::Mat::zeros(diametr, diametr, CV_64FC1);
	}

	for (int i = 0; i < blobs.size(); i++)
	{
		blobs_decompos[i] = PoliMan->Decompose(blobs[i]); 
		Recovery_blobs[i] = PoliMan->Recovery(blobs_decompos[i]);
		ShowBlobDecomposition("Восстановленные цифры:", blobs[i], Recovery_blobs[i]);
		waitKey(0);
	}
	waitKey(0);
	return 0;
}