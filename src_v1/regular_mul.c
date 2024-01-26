#include "matrix_mul.h"
#include <omp.h>
#include <stdlib.h>

#define CAP_SIZE 100

int main(){
	int i=0;
	matrix a,b,ans,y;

	FILE *file_a = fopen("matrices_a.bin", "rb");
    FILE *file_b = fopen("matrices_b.bin", "rb");
    FILE *file_ans = fopen("matrices_ans.bin", "rb");

	while(i<CAP_SIZE){
		i+=1;
		printf("\rMultiplying matrix pair %d", i );

		if(readMatrix(file_a,&a)){
			break;
		}

		if(readMatrix(file_b,&b)){
			break;
		}

		if(readMatrix(file_ans,&ans)){
			break;
		}

		y=matrixMul(a,b);
		if(!compareMatrices(y,ans)){
			return 1;
		}

		
		
	}

	return 0;
}