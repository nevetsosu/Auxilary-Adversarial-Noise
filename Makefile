PORT=8080
all: notebook

notebook:
	jupyter notebook --no-browser --port=$(PORT)
