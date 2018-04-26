# thrift-examples

## Generate

```bash
thrift -out pygen/ --gen py example.thrift
thrift -out jsgen/ --gen js:node example.thrift
```

### Requirements
* Python: `$ pip install thrift`,
* Node.js: `$ npm install thrift`,
* Code generator: [Apache Thrift](https://thrift.apache.org/) (requires only for development).

### Run
* Server: `$ python ./server.py`,
* Python client: `$ ./client.py` or `$ python client.py`,
* Node.js client: `$ ./client.js` or `$ node client.js`.

### Output
```bash
$ python server.py
Starting python server...
Hello from Python!
Hello from JavaScript!
```
```bash
$ python client.py
\\ output
```
```bash
$ node client.js
\\ output
```
