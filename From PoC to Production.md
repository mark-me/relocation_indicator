Hi Barry,

Voor al het kerst en nieuwjaarsgeweld hadden we de verhuisscore besproken, en hoe we deze in productie gaan brengen. 

* We hadden besproken dat we de Relocation score in Python laten draaien omdat het leeuwendeel reeds in Python is gemaakt, en het zonde van de tijd leek het hele proces na te maken in SAS (en wellicht ook vanwege storage issues?). 
* Om er zeker van te zijn dat we geen scores van failliete bedrijven uitleveren, gaat er een pad naar SAS gemaakt worden, om van daaruit de levering van de data te gaan doen. 
* Na afronding van het project met Qualogy, zal ik Joeri 'inweiden' en laten meehelpen  om de kennis te borgen. 

Hieronder beschrijf is de stappen waarmee we de verhuisscore tot product kunnen maken.

1. By end of project with Qalogy
    * Collecting data
		- Location: Laptop 
		- Tool: R
	* Cleaning, transforming and aggregating data
	    - How: by calendar year, all syntax in notebook
	    - Location: Google Cloud 
	    - Tool: Jupyter Notebook
	* Inspecting data
	    - Location: Google Cloud 
	    - Tool: Jupyter Notebook
	* Modelling and scoring
        - Location: Google Cloud 
	    - Tool: Jupyter Notebook

2. From end of project to determination viability of Relocation score, same as above but:
    * Cleaning, transforming and aggregating data
        - How: by 12 month window, syntax made object oriented, requiring less user intervention and better performance.
        - This is now being tested
    * Scoring
        - How: part of the object oriented syntax, requiring no extra user intervention
    * Quality assurance: developing exception reporting to detect anomalies in data and/or scores.
    * Delivery: monthly scored files are uploaded into SAS

3. After the relocation score is determined to be viable:
	* Migrating to Graydon server
	* Integrating collection of data into Python script:
		- Making processing is more centralized 
		- Making R on laptop obsolete, requiring less user intervention

4. The end goal:
	* Doing data loading and processing in the data-store, eliminating the need of user intervention.
	* Automating the process of loading SAS uploading

