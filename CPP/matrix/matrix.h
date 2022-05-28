#pragma once
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <random>
#include <numeric>
#include <functional>
#include <cstdlib>

struct MatrixError : std::runtime_error
{
	MatrixError(const char* error) : std::runtime_error(error) {}
};

class Matrix
{
// private:
public:
	unsigned int m_Rows;
	unsigned int m_Columns;
	std::vector<double> m_Matrix;
	Matrix();
	Matrix(unsigned int rows, unsigned int columns, double initValue = -1);
	Matrix(const Matrix& matrix);
	Matrix(Matrix&& matrix);
	Matrix(const std::vector<double>& data);
	Matrix(const std::vector<double>& data, unsigned int rows, unsigned int columns);
#ifdef _DEBUG
	Matrix(const std::vector<std::vector<double>>& matrix);
#endif // _DEBUG
	Matrix& operator=(const Matrix& matrix);
	Matrix& operator=(Matrix&& matrix);
	~Matrix();

	inline unsigned int GetWidth() const { return m_Columns; }
	inline unsigned int GetHeight() const { return m_Rows; }

	double Sum() const;
	void Randomize(double min = -1, double max = 1);
	void ZeroOut();
	std::vector<double> GetColumnVector() const;
	std::vector<double> GetColumn(unsigned int col);
	std::vector<double> GetRow(unsigned int row);
	void AddRow(std::vector<double> in);
	void SaveMatrix(std::ofstream& outfile) const;

	inline double& operator()(unsigned int row, unsigned int column);
	inline const double& operator()(unsigned int row, unsigned int column) const;
	inline double& operator[](const std::pair<unsigned int, unsigned int>& index);
	inline const double& operator[](const std::pair<unsigned int, unsigned int>& index) const;

	Matrix& operator +=(const Matrix& other);
	Matrix& operator +=(double scalar);
	Matrix& operator -=(const Matrix& other);
	Matrix operator -=(double scalar);
	Matrix operator *=(double scalar);
	Matrix& operator *= (const Matrix& other);
	Matrix operator /= (double scalar);
	Matrix& ElementWise(const Matrix& other);
	Matrix& Transpose();
	Matrix copy(){
		return repeat(1);
	}
	Matrix repeat(int n){
		std::vector<double> t = std::vector<double>(0);
		for(int i=0;i<n;i++){
			for(auto z:m_Matrix)t.push_back(z);
		}
		return Matrix(t, m_Rows, m_Columns);
	}
	template<typename _Func> Matrix& Map(_Func&& func);

	friend std::ostream& operator << (std::ostream& out, const Matrix& m);

	friend Matrix operator+(const Matrix& left, const Matrix& right);
	friend Matrix operator*(const Matrix& matrix, double scalar);
	friend Matrix operator*(double scalar, const Matrix& matrix);
	friend Matrix operator-(const Matrix& left, const Matrix& right);
	friend Matrix operator-(double scalar, const Matrix& matrix);
	friend Matrix operator*(const Matrix& left, const Matrix& right);
	friend Matrix operator/(const Matrix& matrix, double scalar);
	friend Matrix operator/(const Matrix& left, const Matrix& right);

	static Matrix LoadMatrix(std::ifstream& infile);
	static Matrix OneHot(unsigned int one, unsigned int size);
	static Matrix ElementWise(const Matrix& left, const Matrix& right);
	static Matrix Transpose(const Matrix& matrix);
	static Matrix BuildColumnMatrix(unsigned int rows, double value);
	static Matrix Max(const Matrix& first, const Matrix& second);
	template<typename _Func> static Matrix Map(const Matrix& matrix, _Func&& func);
private:
	bool HasSameDimension(const Matrix& other) const;
};

template<typename _Func>
inline Matrix & Matrix::Map(_Func&& func)
{
	std::for_each(m_Matrix.begin(), m_Matrix.end(), [&func](double& x) { x = func(x); });
	return *this;
}

template<typename _Func>
inline Matrix Matrix::Map(const Matrix & matrix, _Func&& func)
{
	Matrix result = matrix;
	result.Map(func);
	return result;
}