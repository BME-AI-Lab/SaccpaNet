# Check List

## Data checking data

### Captures 
1. [x] Every posture have 4 quilt conditions
2. [x] Every quilt condition have 7 posture
3. [x] Every subjects have 7 posture
4. [x] total 150 subjects
5. [x] 96 training subjects
6. [x] 24 validation subject
7. [x] 30 testing subjects

### Joint Annotations 
1. [x] Every (subject*posture) should have same annotaiton
2. [x] Every joints are annotated
3. [ ] For each bone, plot a 2d distribution for distance 
4. [x] For each bone, joints with extreme distance are checked (> sigma)

### Additional check for quilt fusions
1. [ ] Check the explainability for quilt, ie by thickness + gaussian. 
2. [ ] 

### Dataset and data loader
1. [ ] Rewrite for loading from slog
2. [ ] Randomization / Matching should be run in fixed order, ie generate on start of batch
3. [ ] The 



### Network architectures
1. [ ] Remove Layer Norm from the code
2. [ ] Remove Excessive transforms from the code
3. [ ] Draw the network architecutre in svg form
4. [ ] 

### Script runing
1. [ ] Reformat according to runing steps
2. [ ] Move the run generation/ gather code into libary