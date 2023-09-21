# Assessing Climate Change Perceptions and Behaviours in Historical Newspapers 

# Analyse de grandes quantités de textes

There are two different installation methods: 
1. Install Docker and use the provided Docker image 
2. RECOMMENDED ***Install Python 3.x and prepare the work environment, separate installation of the requirements***

## 1. Install Docker and use the provided Docker image

You are about to share local resources (on your workstation) with the container. The container is an autonomous "machine" with its own users. You must therefore make sure that the "others" (*o* for those with little memory resources) have write (*w*), read (*r*) and execution rights on the directories (*x*). The command to use is `chmod`:

```bash
chmod o+rx .
chmod o+r *
```

The Docker container is provided, [available on the hub.docker.com site](https://hub.docker.com/repository/docker/lrucourses/adm-activite-2/tags) (`lrucourses/adm-activite-2:latest`). 

The use of this image is simplified by the `docker-compose` tool that you can use.

First, download the image: `docker-compose pull`. 
Then: `docker-compose build`.
Run: `docker-compose` and then access [`http://127.0.0.1:8888`](http://127.0.0.1:8888) or [`http://localhost:8888`[(http://localhost:8888). Launch only one instance at a time, since the container occupies the local port 8888. 

```
Starting xxx ... done
Attaching to xxx
jupyter_1  | WARN: Jupyter Notebook deprecation notice https://github.com/jupyter/docker-stacks#jupyter-notebook-deprecation-notice.
jupyter_1  | Executing the command: jupyter notebook
jupyter_1  | [I 11:42:55.210 NotebookApp] Writing notebook server cookie secret to /home/XXX/.local/share/jupyter/runtime/notebook_cookie_secret
jupyter_1  | [W 2021-02-28 11:42:56.024 LabApp] 'ip' has moved from NotebookApp to ServerApp. This config will be passed to ServerApp. Be sure to update your config before our next release.
jupyter_1  | [W 2021-02-28 11:42:56.024 LabApp] 'port' has moved from NotebookApp to ServerApp. This config will be passed to ServerApp. Be sure to update your config before our next release.
jupyter_1  | [W 2021-02-28 11:42:56.024 LabApp] 'port' has moved from NotebookApp to ServerApp. This config will be passed to ServerApp. Be sure to update your config before our next release.
jupyter_1  | [W 2021-02-28 11:42:56.024 LabApp] 'port' has moved from NotebookApp to ServerApp. This config will be passed to ServerApp. Be sure to update your config before our next release.
jupyter_1  | [I 2021-02-28 11:42:56.030 LabApp] JupyterLab extension loaded from /opt/conda/lib/python3.8/site-packages/jupyterlab
jupyter_1  | [I 2021-02-28 11:42:56.030 LabApp] JupyterLab application directory is /opt/conda/share/jupyter/lab
jupyter_1  | [I 11:42:56.033 NotebookApp] Serving notebooks from local directory: /work
jupyter_1  | [I 11:42:56.033 NotebookApp] Jupyter Notebook 6.2.0 is running at:
jupyter_1  | [I 11:42:56.033 NotebookApp] http://22d3b09d01ca:8888/?token=1e73a724e063b4e13210b0ba7f1bd85483eaa4b6800e6e7a
jupyter_1  | [I 11:42:56.033 NotebookApp]  or http://127.0.0.1:8888/?token=1e73a724e063b4e13210b0ba7f1bd85483eaa4b6800e6e7a
jupyter_1  | [I 11:42:56.033 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
jupyter_1  | [C 11:42:56.036 NotebookApp] 
jupyter_1  |     
jupyter_1  |     To access the notebook, open this file in a browser:
jupyter_1  |         file:///home/jovyan/.local/share/jupyter/runtime/nbserver-8-open.html
jupyter_1  |     Or copy and paste one of these URLs:
jupyter_1  |         http://22d3b09d01ca:8888/?token=1e73a724e063b4e13210b0ba7f1bd85483eaa4b6800e6e7a
jupyter_1  |      or http://127.0.0.1:8888/?token=1e73a724e063b4e13210b0ba7f1bd85483eaa4b6800e6e7a
```

Open the link that is proposed to you, and you access `jupyter`. Open the `.ipynb` file from `1_text_classification` to start the lab.



---------------------------------------------------

## 2. Install Python 3.x and prepare the work environment, separate installation of the requirements

To write sustainable code, the code must come with a precise definition of its working environment.

### Which Python version?

One major version of Python is currently in use: Python3. If you are starting to learn Python, you have no reason to stick to an old version (2, etc.). 

Start with Python3, this is the future.

How to install Python3 : 

* [for Windows](https://www.python.org/downloads/windows/): you may need to install Python and the packages using the admin CMD.
* For Linux/Mac: Use your package manager

### Choose an Integrated Development Environment (IDE)

* [Atom](https://atom.io/): An open-source text editor developed by GitHub. No specific to Python, extendable with plugins.
* [Spyder](https://pythonhosted.org/spyder/):  built specifically for data science Spyder is a "studio" more than and IDE
* [PyCharm](https://www.jetbrains.com/pycharm/): made by  JetBrains, the same team who made the  Java IDE IntelliJ IDEA. This is a very complete IDE, for those who use Eclipse or MS Studio.
* [Jupyter](http://jupyter.org/install.html): not really an IDE, but an interactive environment in Python. Perfect for starting data science or exploring new data
* for windows [NotePad++](https://notepad-plus-plus.org/fr/)

Try them and choose the one that suits you best. You can always change later or use a different one depending on your task.

### Use an interactive environment: Jupyter

A good way to code in Python interactively is to use Jupyter:

    pip install jupyter
    Jupiter notebook
    
A Jupyter server is now running on your computer, you can access it at http://localhost:8888.

* Create a new notebook: New -> Notebook Python3
* Save your notebook

Then read these documents to learn how to run the code in a notebook : 

* [Notebook Basics](http://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/Notebook%20Basics.ipynb)


### Setup the version control repository

When working in a team (and even alone) the code is shared using a version control tool (SVN, git, mercurial, etc.). Git is now the most popular tool for version control because it is included in most development environments and is provided freely by hosting platforms such as Git Hub, Bitbucket or Gitlab. 

If you are new to git, read : 
* [just a simple guide for getting started with git](http://rogerdudler.github.io/git-guide/)
* [Git tutorials bu Atlassian](https://www.atlassian.com/git/tutorials)


All the code you write will be saved in this repository and committed to the repository. Commit as soon as you have something working, you completed the answer to a question or when you stop working on the project for a moment.

When you create a new file, you first have to add the file to the repository : 

	git add the_new_file.py
	
**Never** use

	git add.
	
it would add all the files in the current directory to git, including hidden files, images, files in ENV, etc...


To register the changes of a file to the local repository, use `git commit`. Provide a useful commit message, for example :

	git commit -m "extract features from images" 
	git commit -m "fix bug refs #197"
	git commit -m "clustering algorithm in user profiles: WIP"
	
These messages are useless and unprofessional: 

	git commit -m "work"
	git commit -m "fix"
	
When all the changes are committed to the local repository, send your modification to the server : 

	git push


### Virtual environnement

As a data scientist, you will be working on different projects, possibly at the same time, but for different clients, with different environmental constraints. One client could use Python2 and the other Python3, one client could use Python3.5 and the other Python3.6. The syntax and functions API may not be compatible between different versions of Python or packages and you can not install different versions of the same package on your system.
It means that for every project, you should create an independent Python environment, in which you can install the versions of Python and packages requested by the client. This environment is called a virtual environment.

The first way to create a virtual environment is to use [virtualenv](https://virtualenv.pypa.io/en/stable/)

Create a virtual environment named ENV : 

	$  virtualenv ENV
    
If you have more than one Python version installed on your computer, you can specify the Python version :

	virtualenv -p python3 ENV
	virtualenv -p C:\full\path\python3 ENV

Activate the virtual environment : 

    $ source ENV/bin/activate
    
and for Windows : 

    $ \path\to\env\Scripts\activate

The Python interpreter is now the one local to the virtual environment : 

	(ENV) $ which python
	/home/user/my-project/ENV/bin/python

You can now install packages for your local version of Python without interfering with your other projects.

	pip install pandas pillow
	pip install -r requirements.txt
	
To deactivate the environment : 

    deactivate

A second way to create a virtual environment is to use a conda environment. This is only possible if you use conda. 

    conda create -n myenv python=3.4
    source activate myenv
    source deactivate


It is not recommended to mix conda env and virtualenv

Note: Do not commit virtual env files in the git repository. To avoid any problem, create a .gitignore file containing `ENV` and commit it. Update this file as needed.

See : 
* [Atlassian git ignore tutorial](https://www.atlassian.com/git/tutorials/gitignore)
* [Git Community Book : Ignorer les fichiers ](http://alexgirard.com/git-book/intermediaire/git-ignore/)



### Requirements

When working with a team on a project, every member of the team must be able to run the code very easily without struggling with installation problems. Since Data Science is a very rapidly evolving field, the Python packages are updated regularly and from time to time, the syntax for using a module or a function changes. Therefore, your code must come with a precise definition of the packages and package version needed to make it run smoothly. 

Since you are working in a Python virtual environment, the Python and package version is specific to your current project. You can print this information with the pip command `freeze`.

For example : 

    (ENV) $ pip freeze
	numpy==1.13.1
	olefile==0.44
	pandas==0.20.3
	python-dateutil==2.6.1
	pytz==2017.2
	scikit-learn==0.19.0
	scipy==0.19.1
	six==1.11.0
	tqdm==4.17.1

This information can be saved in a file :

    (ENV) $ pip freeze > requirements.txt
    
and used to setup a running environment : 

	$ virtualenv -p python3 ENV
    $ source ENV/bin/activate
    (ENV) $ pip install -r requirements.txt
    
 Commit requirements.txt to your repository (it should be empty for now).

## Best practices for python code

A python program is usually not made to be developed by only one developer and executed only once. It is meant to be shared and used many times, in different conditions and it should be designed and written with these goals in mind.

### Language and encoding

Programming is an international activity and if your native language is not English, there is a great chance that your code will be read or used by someone who does not speak or read your language. So **write your code in English**: variable and function names, comments, and documentation.

Your code is also expected to work in an international environment, supporting strings in any language: all your **files must be encoded in UTF-8**. Add this line at the beginning of your code : 

    # -*- coding: utf-8 -*-

See [PEP0263](https://www.python.org/dev/peps/pep-0263/) for details.

### Python programming style

A coherent style increases the legibility and eases the bug-tracking process. A writing style has been defined for Python programs: the [PEP8](http://pep8.org/). Your code must be **pep8 compliant**.

Tools can help you to verify that your code is pep8 compliant : [pylint](https://www.pylint.org/). Add a plugin to your favourite IDE to check that your code is PEP8 compliant while writing it.

See this [example](https://gist.github.com/RichardBronosky/454964087739a449da04) of code in PEP8.

### Command line

When programming a Python script, the parameters (input data, output data, script behaviours) must be defined using the command line options and arguments. This is both a documentation of the usage of the script and a way to check that the user calls the script with the correct arguments. See for example [argparse](https://docs.python.org/3.6/howto/argparse.html#id1), the recommended command-line parsing module in the Python standard library
### Logging

When programming, it is natural to check the results of the program at different stages. It is tempting to display values of variables using `print`, keeping in mind that all the `print` should be removed before sending the code to production. But as the size of the program grows, there is a great chance that a few of the `print` will go to production and create bugs difficult to locate: log file filling the /tmp disk, runtime error due to printing utf8 characters to a device not supporting utf8, etc.

One clean way of printing information as the program runs is to use a Python logging system : 

	import logging
	logging.warning('Watch out!')  # will print a message to the console
   
Several levels of information can be defined :

- DEBUG: Detailed information, typically of interest only when diagnosing problems.
- INFO: Confirmation that things are working as expected.
- WARNING: An indication that something unexpected happened, or indicative of some problem shortly (e.g. ‘disk space low’). The software is still working as expected.
- ERROR: Due to a more serious problem, the software has not been able to perform some functions. CRITICAL: A serious error, indicating that the program itself may be unable to continue running

Logging messages can also be sent to a file : 

    import logging
    logging.basicConfig(filename='example.log',level=logging.DEBUG)
    
See this [tutorial](https://docs.python.org/3/howto/logging.html#logging-basic-tutorial) and the [documentation](https://docs.python.org/3/library/logging.html) for more information. 
