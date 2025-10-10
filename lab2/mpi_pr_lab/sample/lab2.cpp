#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <vector>

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}

	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long total_pixels = 0;

	// 計算所有x值從0到r-1的像素點
	for (unsigned long long x = 0; x < r; x++) {
		unsigned long long y = ceil(sqrtl(r*r - x*x));
		total_pixels += y;
		total_pixels %= k;
	}

	printf("%llu\n", (4 * total_pixels) % k);
	return 0;
}
