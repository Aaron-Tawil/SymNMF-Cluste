1. k=3, max_iter = 600
2. k=7, max_iter = not provided
3. k=15, max_iter = 300

4. k=2, max_iter = 2
4. k=7, max_iter = 999

	C-x+VALGRIND 4. k=8, max_iter = 2 -> invalid clusters
	C-x 4. k=1, max_iter = 2 -> invalid clusters
	C-x 4. k=-2, max_iter = 2 -> invalid clusters

	C-x+VALGRIND 4. k=2, max_iter = 1000 -> invalid maxIter
	C-x 4. k=2, max_iter = 1 -> invalid maxIter
	C-x 4. k=2, max_iter = -2 -> invalid maxIter
	C-x 4. k=2, max_iter = not provided -> invalid maxIter
