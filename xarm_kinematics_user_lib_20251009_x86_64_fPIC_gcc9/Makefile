
all:
	g++ -std=c++11 main_xarm.cpp libkinematics_xarm_user.a -o main_xarm -fopenmp
	g++ -std=c++11 main_lite6.cpp libkinematics_xarm_user.a -o main_lite6 -fopenmp
	g++ -std=c++11 main_uf850.cpp libkinematics_xarm_user.a -o main_uf850 -fopenmp
clean:
	rm main_xarm 
	rm main_lite6
	rm main_uf850
