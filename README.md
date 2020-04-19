# Datarevenue Code Challenge

Congratulations for making it to the Data Revenue Code Challenge 2020. This coding challenge will be used to evaluate your technical as well as your communication skills.

You will need docker an docker-compose to run this repository: 

* [How to install docker](https://docs.docker.com/engine/installation/)
* [How to install docker-compose](https://docs.docker.com/compose/install/)

## Goals
The repository you see here is a minimal local version of our usual task orchestration pipeline. We run everything in docker containers. So each task must expose its functionality via a CLI. We then use luigi to spin up the containers and pass the necessary arguments to each container. See more details [here](https://www.datarevenue.com/en/blog/how-to-scale-your-machine-learning-pipeline).

The repository already comes with a leaf task implemented which will download the data set for you.

The goal of this challenge is to implement a complete machine learning pipeline. This pipeline should build a proof of concept machine learning model and evaluate it on a test data set.

**An important part of this challenge is to assess and explain the model to a fictional client with limited statistical knowledge. So your evaluation should include some plots on how your model makes the predictions. Finally you need to give an essesment if it will make sense for the client to implement this model!**

### Challenge
To put things into the right perspective consider the following fictional scenario: 

You are a AI Consultant at Data Revenue. One of our clients is a big online wine seller. After a successful strategic consulting we advice the client to optimize his portfolio by creating a **rating predictor (predict points given to a wine) for his inventory.** We receive a sample dataset (10k rows) from the client and will come back in a week to evaluate our model on a bigger data set that is only accessible from on-premise servers (>100k rows).

The task is to show that a good prediction is possible and thereby make it less risky to implement a full production solution. Our mini pipeline should later be able to run on their on premise machine which has only docker and docker-compose installed.

### Data set

Here is an excerpt of dataset you will be working on:

country|description|designation|points|price|province|region_1|region_2|taster_name|taster_twitter_handle|title|variety|winery
---|---|---|---|---|---|---|---|---|---|---|---|---
Italy|Fragrances suggest hay, crushed tomato vine and exotic fruit. The bright but structured palate delivers peach, papaya, cantaloupe and energizing mineral notes alongside fresh acidity. It's nicely balanced with good length,|Kirchleiten|90|30.0|Northeastern Italy|Alto Adige||Kerin O’Keefe|@kerinokeefe|Tiefenbrunner 2012 Kirchleiten Sauvignon (Alto Adige)|Sauvignon|Tiefenbrunner
France|Packed with fruit and crisp acidity, this is a bright, light and perfumed wine. Red-berry flavors are lifted by red currants and a light spice. Drink now for total freshness.||87|22.0|Loire Valley|Sancerre||Roger Voss|@vossroger|Bernard Reverdy et Fils 2014 Rosé (Sancerre)|Rosé|Bernard Reverdy et Fils
Italy|This easy, ruby-red wine displays fresh berry flavors and a light, crisp mouthfeel. Pair this no-fuss wine with homemade pasta sauce or potato gnocchi and cheese.||86||Tuscany|Chianti Classico||||Dievole 2009  Chianti Classico|Sangiovese|Dievole
US|Pretty in violet and rose petals this is a lower-octane Pinot Noir for the winery. Exquisitely rendered in spicy dark cherry and soft, supple tannins, it hails from a cool, coastal vineyard site 1,000 feet atop Occidental Ridge, the coolest source of grapes for Davis.|Horseshoe Bend Vineyard|92|50.0|California|Russian River Valley|Sonoma|Virginie Boone|@vboone|Davis Family 2012 Horseshoe Bend Vineyard Pinot Noir (Russian River Valley)|Pinot Noir|Davis Family
US|This golden wine confounds in a mix of wet stone and caramel on the nose, the body creamy in vanilla. Fuller in style and body than some, it remains balanced in acidity and tangy citrus, maintaining a freshness and brightness throughout. The finish is intense with more of that citrus, plus an accent of ginger and lemongrass.|Dutton Ranch|93|38.0|California|Russian River Valley|Sonoma|Virginie Boone|@vboone|Dutton-Goldfield 2013 Dutton Ranch Chardonnay (Russian River Valley)|Chardonnay|Dutton-Goldfield
US|This is a lush, rich Chardonnay with especially ripe pineapple, peach and lime flavors, as well as a coating of oaky, buttered toast.|Signature Selection|84|14.0|California|Dry Creek Valley|Sonoma|||Pedroncelli 2012 Signature Selection Chardonnay (Dry Creek Valley)|Chardonnay|Pedroncelli
US|Intensely aromatic of exotic spice, potpourri and dried fig, this dry Gewürztraminer is a bit atypical, but thought provoking and enjoyable. Lemon and apple flavors have a slightly yeasty tone, but brisk acidity and puckering tea-leaf tannins lend elegance and balance.|Spezia|87|25.0|New York|North Fork of Long Island|Long Island|Anna Lee C. Iijima||Anthony Nappa 2013 Spezia Gewürztraminer (North Fork of Long Island)|Gewürztraminer|Anthony Nappa
US|Dry, acidic and tannic, in the manner of a young Barbera, but the flavors of cherries, blackberries and currants aren't powerful enough to outlast the astringency. Drink this tough, rustic wine now.||84|35.0|California|Paso Robles|Central Coast|||Eagle Castle 2007 Barbera (Paso Robles)|Barbera|Eagle Castle
France|Gold in color, this is a wine with notes of spice, rich fruit and honey, which are all surrounded by intense botrytis. This is a wine that has great aging potential, and its superripeness develops slowly on the palate.||94||Bordeaux|Sauternes||Roger Voss|@vossroger|Château Lamothe Guignard 2009  Sauternes|Bordeaux-style White Blend|Château Lamothe Guignard
France|Steel and nervy mineralogy are the hallmarks of this wine at this stage. It's still waiting for the fruit to develop, but expect crisp citrus and succulent apples. The aftertaste, tensely fresh now, should soften as the wine develops. This 90% Sauvignon Blanc and 10% Sémillon blend comes from the estate's small vineyard on the slope near Cadillac.||88|12.0|Bordeaux|Bordeaux Blanc||Roger Voss|@vossroger|Château Boisson 2014  Bordeaux Blanc|Bordeaux-style White Blend|Château Boisson


### Prerequisites
Before starting this challenge you should know:
1. How to train and evaluate a ML model.
1. Have solid understanding of the [pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html) library and ideally the [dask](http://docs.dask.org/en/latest/dataframe.html) parallel computing library.
1. How to run [docker containers](https://docs.docker.com/get-started/).
1. How to specify tasks and dependencies in Spotify's [luigi](https://luigi.readthedocs.io/en/stable/example_top_artists.html).
1. Have read our [TaC blogpost](https://www.datarevenue.com/en/blog/how-to-scale-your-machine-learning-pipeline). This will be very helpful to understand this repo's architecture!

### Requirements

To specify requirements better let's break this down into individual tasks.

#### 1. DownloadData
We already got you covered and implemented this task for you.

#### 2. Make(Train|Test)Dataset
We supply you with the scaffold for this task, so you can start and explore dask or simply go ahead with you usual pandas script.

Read the csv provided by DownloadData and transform it into a numerical matrix ready for your ML models. 

Be aware that the dataset is just a sample from the whole dataset so the values in your columns might not represent all possible values. 

At Data Revenue we use dask to parallelize Pandas operations. So we include also a running dask cluster which you *can* (you don't need to) use. Remember to partition your csv if you plan on using dask (by using [blocksize](http://docs.dask.org/en/latest/dataframe-api.html#dask.dataframe.read_csv)).

Don't forget to split your data set according to best practices. So you might need more than a single task for this.

#### 3. TrainModel
Choose a suitable model type and train it on your previously built data set. We like models that don't take forever to train. Please no DNN (this includes word2vec). For the sake of simplicity you can use fixed hyperparameters (hopefully "hand tuned"). Serialize your model to a file. If necessary this file can include metadata. 

The final data set will have more than 100k rows.

#### 4. EvaluateModel
Here you can get creative! Pick a good metric and show your communication and presentation skills. Load your model and evaluate it on a held out part of the data set. This task should have a concrete outcome e.g. a zip of plots or even better a whole report (check the [pweave](http://mpastell.com/pweave/) package).

**You will most likely need the output of this task to tell the client if the model is suited for his endavour. This should include a assesment of the quality of the model, and also the consequences of the errors that the model makes.**

#### Other requirements
- Each task:
    - Needs to be callable via the command line
    - Needs to be documented
    - Should have **single** file as output (if you have two consider putting them into a single file or use a .SUCCESS flag file as the tasks output)
- Task images that aren't handled by docker-compose should be build and tagged in `./build-task-images.sh`
- Task images should be minimal to complete the task
- The data produced by your tasks should be structured (directories and filename) sensibly inside `./data_root`
- Don't commit anything in `./data_root`, use `.gitignore`
- Your code should be PEP8 conform


## Get Started

To get started execute the DownloadData task we provide this task already completely containerized for you. Let's first build the images, we have included a script so this is more streamlined:

`./build-task-images.sh 0.1`

Now to execute the pipeline simply run: 

`docker-compose up orchestrator` 

This will download the data for you. It might be a good idea to execute: 

`watch -n 0.1 docker ps`

in a different terminal window to get a sense of what is going on. 

We recommend to start developing in notebooks or you IDE locally if you're not very familiar with docker. This way we can consider your solution even if you don't get the whole pipeline running. Also don't hesitate to contact us if you hit a serious blocker instead of wasting too much time on it.

### NOTE: Configure your docker network

Docker runs containers in their own networks. Compose automatically creates a network
for each project. This project assumes that this network is named 
`code-challenge-2020_default` depending on your folder name and compose version this 
might not always be the case. You will get an error when trying to download the data if
this network is named differently for you. If you run into this error, please execute:
`docker network ls` and identify the correct network name. Next open the 
`docker-compose.yml` and edit the env variable on the orchestrator service.

### Troubleshooting in Task Containers
We also included a Debug task for you which you may start if you need a shell
inside a task's container. Make sure to adjust the correct image if you want to 
debug a task other then DownloadData. Then run:

`docker-compose run orchestrator luigi --module task Debug --local-scheduler`

this will spawn a task with luigi but set it to sleep for 3600 seconds. You can
use that time to get a shell into the container, but first you need to find 
the containers name, so from a different terminal run:

`docker ps`

check for a container named `debug-<something>` then execute

`docker exec -ti debug-<something> shell`

Now you're in the container and can move around the filesystem execute commands 
etc. To exit simply type `exit`

### Exposed Dashboards
This scaffold exposes 2 dashboards:
- dask-scheduler @ [http://localhost:8787](http://localhost:8787). This let's you view how dask is executing your computation graph find more out [here](http://docs.dask.org/en/latest/diagnostics-distributed.html#dashboard)!
- luigi-scheduler @ [http://localhost:8082](http://localhost:8082). This shows you you're high level task progress.  

## Evaluation Criteria
Your solution will be evaluated against following criteria:

* Is it runnable? **25 points**
* ML Best Practices **20 points**
* Presentation of results (during interview) **20 points**
* Code Quality (incl. Documentation and PEP8) **10 points**
* Structure/Modularity **10 points**
* Correct use of linux tools (dockerfiles, shellscripts) **10 points**
* Performance (concurrency, correct use of docker cache) **5 points**

## Task as Container TLDR;
This is a TLDR; of [TaC blogpost](https://www.datarevenue.com/en/blog/how-to-scale-your-machine-learning-pipeline)

- We spawn containers from a orchestrator container.
- These spawned container run pipeline steps.
- Services that need to be accessed by the containers are built and managed via docker-compose.
- We see the orchestrator as a service.
- To share data between containers we must tell the orchestrator where our project is located on the host machine. The orchestrator will then mount this directory into `/usr/share/data` in dynamically spawned containers. 
- To allow the orchestrator to spawn containers we must expose the hosts docker socket to it.

## FAQ

> Can I use notebooks?

Yes you are encouraged to use notebooks to do ad-hoc analysis. Please include them in your submission. Though having a pipeline set up in a notebook does not free you from submitting a working containerized pipeline.

> What is the recommended way to develop this?

Just install all the needed packages in a conda-env or virtualenv and start developing in you favorite IDE or within the beloved jupyter notebook or both. Once you are happy with the results, expose your notebooks functionality in a CLI and package it with a Dockerfile. 

> Can I use other technologies? Such as R, Spark, Pyspark, Modin, etc.

Yes you can as long as you can provision the docker containers and spin up all the necessary services with docker-compose.

> Do you accept partial submissions?

Yes you can submit you coding challenge partially finished in case you don't finish in time or have trouble with all the docker stuff. Unfinished challenges will be reviewed if some kind of model evaluation report is included (notebook or similar). You will lose points though as it will be considered as not runnable (no points in runnable category, no points in linux tools category and maximum 3 points in performance category).

> I found a bug! What should I do?

Please contact us! We wrote this in a hurry and also make mistakes. PRs on bugs get you extra points ;)

> I have another question!

Feel free to create an issue! Discussions in issues are generally encouraged.


## Submission
Please zip your solution including all files and send to us with
the following naming schema:
```
cc19_<first_name>_<last_name>.zip
```
