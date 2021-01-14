#pragma once
#ifndef GF_H
#define GF_H
#include <vector>


static void Invert(std::vector<float>& x)
{
	for (int i = 0; i < x.size(); i++)
		x[i] = -x[i];
}

//[0][1][2]
//[3][4][5]
//[6][7][8]

static bool Win(std::vector<float> x)
{
	if ((x[0] == 1) && (x[1] == 1) && (x[2] == 1))
		return true;
	if ((x[3] == 1) && (x[4] == 1) && (x[5] == 1))
		return true;
	if ((x[6] == 1) && (x[7] == 1) && (x[8] == 1))
		return true;
	if ((x[0] == 1) && (x[3] == 1) && (x[6] == 1))
		return true;
	if ((x[1] == 1) && (x[4] == 1) && (x[7] == 1))
		return true;
	if ((x[2] == 1) && (x[5] == 1) && (x[8] == 1))
		return true;
	if ((x[0] == 1) && (x[4] == 1) && (x[8] == 1))
		return true;
	if ((x[2] == 1) && (x[4] == 1) && (x[6] == 1))
		return true;
	return false;
}
#endif // !GF_H