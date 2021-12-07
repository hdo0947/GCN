# GCN
Graph Covolutional Network C implementation

1) Clone the repository:
```bash
git clone https://github.com/hdo0947/GCN.git
```


2) CD to the repo directory:
```bash
cd GCN
```

3) Prepare dataset, download the DCN_data from https://drive.google.com/file/d/1kVhzUskwzco9gJNxyLAOx_wS7Q3wOWb5/view?usp=sharing, and then unzip it to your </PATH/TO/DCN>

4) Build the executable code:
```bash
./build
```

5) Add library path:
```bash
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/_build/
export LD_LIBRARY_PATH
```

6) Run the GCN:
```bash
./_build/GCN cora
# or
./_build/GCN citeseer
```
