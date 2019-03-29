# Setup
## Install java8 (if it is not installed)

```{bash}
brew tap caskroom/versions
brew cask install java8
```
* If this path exist, then skip this step.
    
      cd /Library/Java/JavaVirtualMachines/jdk1.8.0_202.jdk/
[This is not helpful!](https://www.oracle.com/technetwork/java/javase/downloads/jdk12-downloads-5295953.html)

## Install pyspark
```{python}
pip install pyspark
```
## Open jupyter notebook
Copy this in, it helps in my labtop
```{python}
from pyspark import SparkContext
import os 

os.environ['JAVA_HOME']="/Library/Java/JavaVirtualMachines/jdk1.8.0_202.jdk/Contents/Home/"

os.environ['PYSPARK_SUBMIT_ARGS'] = "--master local[2] pyspark-shell"
```
