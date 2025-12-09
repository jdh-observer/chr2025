---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.6
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"tags": ["title"]}

# Browse a Factgrid RDF dump in a Jupyter notebook

+++ {"tags": ["contributor"]}

### Mattia Bunel[![orcid](https://orcid.org/sites/default/files/images/orcid_16x16.png)](https://orcid.org/0000-0001-7751-3507) 
geocitegeom

+++ {"tags": ["contributor"]}

### Elisabeth Guerard[![orcid](https://orcid.org/sites/default/files/images/orcid_16x16.png)](https://orcid.org/0000-0001-7742-4141) 
c2dh

+++ {"tags": ["contributor"]}

### Konrad Hinsen[![orcid](https://orcid.org/sites/default/files/images/orcid_16x16.png)](https://orcid.org/0000-0003-0330-9428) 
cnrsrfrr

+++ {"tags": ["contributor"]}

### Raphaëlle Krummeich[![orcid](https://orcid.org/sites/default/files/images/orcid_16x16.png)](https://orcid.org/0000-0002-6170-8243) 
idees

+++ {"tags": ["contributor"]}

### Sébastien Rey-Coyrehourcq[![orcid](https://orcid.org/sites/default/files/images/orcid_16x16.png)](https://orcid.org/0000-0001-7296-9695) 
idees

+++

# Introduction and Workshop Scope : "RDF Dump"

The workshop offers a playground notebook to create or to explore RDF dumps. If you are unfamiliar with linked data, we recommend reading the [corresponding lesson of "Programming Historian"](https://doi.org/10.46430/phen0068).

This workshop offers a brief practical introduction of how to browse RDF dumps or how to build RDF graphs for database exploration and open research purposes. Basic know-how about querying or building LOD data, basic analysis for exploration, here in the Python language, and editorial issues should be in the hands of trainees.

This workshop is split in five parts, plus further reading:
1. A first part introduces Python packages to be used in Jupyter notebooks. Some ressource gives further tutorials for working on your local machine.
2. A second part provides Python recipies for browsing dumps from wikidata and Factgrid wikibase, in order to collect data from the graph.
3. A third part helps with building RDF graphs in Python using `rdflib`.
4. A fourth part shows how to use of SPARQL queries and how to display results on maps or timelines, leading to first concrete examples.
5. A last part aims at producing a scientific publication within the Journal of Digital History's editorial technology stack.

# General framework: from data to scientific publication

## From data collection

Data in the scope of the workshop are linked open data available on FactGrid, a database for historical research. It is an international collaborative platform using Wikimedia's ground-breaking wikibase software, run at the Gotha Research Centre and hosted at the Thuringian State and University Library ThULB in Jena (see https://blog.factgrid.de/welcome).







```{code-cell} ipython3
:tags: [figure-factgrid-*]

from IPython.display import IFrame
IFrame('https://blog.factgrid.de/', width='100%', height='422')
```

## Data model under scope

These data follow a general modelling framework for RDF triplets with domains and ranges that can take many value types (statements, xsd date, simple values etc.), like in a wikidata dump. Predicates also follow the wikidata schema with differentiating paths regarding the type of property under scope (see for exemple, figure below).

```{code-cell} ipython3
:tags: [figure-wiki-*]

from IPython.display import Image, display
metadata={
    "jdh": {
        "module": "object",
        "object": {
            "type":"image",
            "source": ["Wikidata model, source : [https://www.mediawiki.org/wiki/Wikibase/Indexing/RDF_Dump_Format](https://www.mediawiki.org/wiki/Wikibase/Indexing/RDF_Dump_Format)"]
        }
    }
}
display(Image("./img/Rdf_mapping-vector.svg.png"),metadata=metadata)
```

## Computational notebook framework: multi-purpose from exploration to scientific publication, through workshop training

Within this workshop, you will be invited to play with your own computational Jupyter notebook online. However, the more curious or skilled may follow our guidelines (in French) to install local Jupyter instances for running the workshop's computational notebooks directly on your own machine: [Ressources to install Jupyter Notebook framework on your local machine](https://mise-en-pratique-5e5223.gricad-pages.univ-grenoble-alpes.fr/jupyter.html).

The Jupyter notebook may take two different forms: one based on mystmd technology, and the traditional JSON format ipynb. If you wish to prepare a JDH article proposal with the framework proposed, you may follow these two steps:
- push your ipynb into a github repo,
- copy-paste the URL of your ipynb computational notebook on your github repo there: [View JDH](https://journalofdigitalhistory.org/en/notebook-viewer-form)

```{code-cell} ipython3
:tags: [figure-jdh-*]

from IPython.display import Image, display
metadata={
    "jdh": {
        "module": "object",
        "object": {
            "type":"image",
            "source": ["Narrative, hermeneutical and code layers of the Journal of Digital History, image captured from the JDH article [https://doi.org/10.1515/JDH-2023-0018](https://doi.org/10.1515/JDH-2023-0018)"]
        }
    }
}
display(Image("./img/jdh.png"),metadata=metadata)
```

+++ {"tags": ["hermeneutics"]}

# Technologies to compute: Python libraries to perform SPARQL queries

For this workshop, we chose the Python language and librairies for SPARQL queries and data visualization.

## Python packages used

- `rdflib`: working with RDF data and making SPARQL queries
- `networkx`: managing graphs
- `matplotlib`: plotting
- `plotly`: plotting as well
- `pandas`: dataframes (tabular data) in Python
- `geopandas`: an extension of `pandas` for spatial objects
- `ipysigma`: for display interactives graphs

## A "reproducible" process ?

Another package is used here to give information about how we produced the computational document. Notably, the package `session_info` gives information about system and package versions, enabling reproducibility in a controlled computational environment.

- `session_info` : retrieve software version numbers

## Install packages

In your terminal or inside your notebook, you can install these packages with `pip`:

```{code} python
!pip install session_info rdflib networkx matplotlib plotly pandas ipysigma
```

# Ontologies, namespaces or vocabularies: before browsing FactGrid wikibase dump

As stated in programming historian's lesson, building knowledge graphs require some vocabularies, like w3c standards. These namespaces are url were classes or concepts may be found within their hierarchical relations or inherited relations or equivalence (sameAs), for example. SPARQL syntax usually shorten these namespaces with associated prefixes, stated at the head of the SPARQL query.

## Namespaces and prefixes

Many widely used namespaces are predefined in `rdflib`, e.g. `FOAF`, `DublinCore`, `Schema`, or `XSD`.
Many widely used namespaces are predefined in `rdflib` package, e.g. `FOAF`, `DublinCore`, `Schema`, or `XSD`. To have a complete overview of namespaces directly available within the proposed `rdflin` framework, the following steps are useful :

```{code-cell} ipython3
from rdflib import Graph, Namespace
# rdflib knows about quite a few popular namespaces, like W3C ontologies, schema.org etc.
#from rdflib.namespace import FOAF , XSD

# Create a Graph
g = Graph()


for prefix,ns in g.namespaces():
   print(f"""{prefix}: {ns}""")
```

However, we see no `wikibase`, nor `wikidata` or `dbpedia` namespaces. These have to be declared and bounded to specific prefixes.

## Querying Wikidata

```{code-cell} ipython3
# import Graph & Namespace

from rdflib import Graph, Namespace
from rdflib.namespace import NamespaceManager

# Wikidata namespace
WD = Namespace("http://www.wikidata.org/entity/")

# define graph to be crawled in
g2 = Graph()
g2.bind("wd", WD)

# define query about Mona Lisa QID
qres = g2.query(
  """
  SELECT ?o
  WHERE {
    SERVICE <https://query.wikidata.org/sparql> {
      wd:Q12418 rdfs:label ?o .
    }
  }
  LIMIT 10
  """
)
```

```{code-cell} ipython3
# print results nicely
for row in qres:
   print("wikidata identifier Q12418 <has label> : %s" %row)
```

# Querying FactGrid

## 1.0 - Collecting keywords

```{code-cell} ipython3
# test with Factgrid namespace
FG_WD = Namespace("https://database.factgrid.de/entity/")
FG_WDT = Namespace("https://database.factgrid.de/prop/direct/")

# define graph to be crawled in
g3 = Graph()
g3.bind("fg_wd", FG_WD)
g3.bind("fg_wdt", FG_WDT)

# define query for keywords
qres = g3.query(
"""
SELECT DISTINCT ?p (COALESCE(?p_labell,'') AS ?p_label)
WHERE {
  SERVICE <https://database.factgrid.de/sparql> {
    ?p fg_wdt:P1132 fg_wd:Q960698.
    OPTIONAL {
      ?p rdfs:label ?p_labell.
      FILTER(lang(?p_labell) IN ('en'))
    }
  }
}
ORDER BY ?p
"""
)
```

```{code-cell} ipython3
# print results nicely
for row in qres:
   print(row.asdict()['p_label'])
```

## TODO: classes of keywords ?

## 2.0 - Collecting Factgrid projects names etc.

```{code-cell} ipython3
FG_P = Namespace("https://database.factgrid.de/prop/")
FG_PS = Namespace("https://database.factgrid.de/prop/statement/")
g4 = Graph()
g4.bind("fg_wd", FG_WD)
g4.bind("fg_wdt", FG_WDT)
g4.bind("fg_p", FG_P)
g4.bind("fg_ps", FG_PS)
# define query for projects
qres = g4.query(
   """
SELECT ?author_labell ?project_labell ?date ?localisation_labell ?coordinates
WHERE {
    SERVICE <https://database.factgrid.de/sparql> {
?project fg_wdt:P2 fg_wd:Q11295.
   OPTIONAL {
      ?project rdfs:label ?project_labell.
      FILTER(lang(?project_labell) IN ('en'))
   }
   OPTIONAL  {
      ?project fg_wdt:P49 ?date .
   }
   OPTIONAL {
      ?project fg_p:P297/fg_ps:P297 ?localisation .
      ?localisation rdfs:label ?localisation_labell.
      FILTER(lang(?localisation_labell) IN ('en'))
      ?localisation fg_wdt:P48 ?coordinates
    }
   OPTIONAL {
      ?project fg_wdt:P21 ?author .
      ?author rdfs:label ?author_labell
      FILTER(lang(?author_labell) IN ('en'))
          }
}
}
ORDER BY ?project
   """
)
```

## [TOOL] Create a function to convert from SPARQLResult to Dataframe

```{code-cell} ipython3
from pandas import DataFrame
from rdflib.plugins.sparql.processor import SPARQLResult

def sparql_results_to_df(results: SPARQLResult) -> DataFrame:
    """
    Export results from an rdflib SPARQL query into a `pandas.DataFrame`,
    using Python types. See https://github.com/RDFLib/rdflib/issues/1179.
    """
    return DataFrame(
        data=([None if x is None else x.toPython() for x in row] for row in results),
        columns=[str(x) for x in results.vars],
    )
```

```{code-cell} ipython3
:tags: [table-sparkle-*, data-table]

qres_df = sparql_results_to_df(qres)
qres_df
```

## 2.0 Map of location of projects (proportional circles ? heat map ?)

```{code-cell} ipython3
---
jdh:
  module: object
  object:
    source: [Location of project]
    type: image
tags: [figure-map-*]
---
import geopandas as gpd

# We create a spatial dataframe with geopandas
gs = gpd.GeoSeries.from_wkt(qres_df['coordinates'])
# CRS: Coordinates Refrence System
gdf = gpd.GeoDataFrame(qres_df, geometry=gs, crs="EPSG:4326")

gdf.explore()
```

## 2.1 Querying the number of statements per projects

```{code-cell} ipython3
FG_WIKIBASE = Namespace("http://wikiba.se/ontology#")
g5 = Graph()
g5.bind("fg_wd", FG_WD)
g5.bind("fg_wdt", FG_WDT)
g5.bind("fg_wikibase", FG_WIKIBASE)
# define query for projects
qres = g5.query(
   """
SELECT ?project_labell ?date ?stmtcount
WHERE {
    SERVICE <https://database.factgrid.de/sparql> {
        ?project fg_wdt:P2 fg_wd:Q11295 .
	?project fg_wdt:P49 ?date .
	?project fg_wikibase:statements ?stmtcount .
      	?project rdfs:label ?project_labell.
      	FILTER(lang(?project_labell) IN ('en'))
   }
}
ORDER BY DESC (?stmtcount)
LIMIT 100
   """
)
```

```{code-cell} ipython3
---
jdh:
  module: object
  object:
    source: [Number of statemzents per project]
    type: image
tags: [table-sparkle-project-*, data-table]
---
sparql_results_to_df(qres)
```

## Build query for project as domain and as range

```{code-cell} ipython3
g8 = Graph()
g8.bind("fg_wd", FG_WD)
g8.bind("fg_wikibase", FG_WIKIBASE)
# define query for projects
qres = g8.query(
   """
SELECT ?s ?wd ?o
WHERE {
  SERVICE <https://database.factgrid.de/sparql> {
    {
	  BIND (fg_wd:Q467586 as ?s)
	  ?s ?p ?o.
	  ?wd fg_wikibase:claim ?p.
	}
	UNION
    {
	  BIND (fg_wd:Q467586 as ?o)
	  ?s ?p ?o.
	  ?wd fg_wikibase:claim ?p .
	}
}}
"""
)

# Due of rdflib's limitation
for row in qres:
    g8.add(row)

print(g8.serialize(format='n3'))
```

## Draw resulting graph

```{code-cell} ipython3
---
jdh:
  module: object
  object:
    source: [Network]
    type: image
tags: [figure-network-bis-*]
---
from rdflib.extras.external_graph_libs import rdflib_to_networkx_graph
import networkx as nx
import matplotlib.pyplot as plt

ug8 = rdflib_to_networkx_graph(g8)
nx.draw(ug8)
```

We can also use the ipysigma library, for displaing an interactive graph (but its work only in an interactive jupyter session)

```{code-cell} ipython3
from ipysigma import Sigma
Sigma(ug8)
```

## 2.2 Querying for the list of statements for a given project

A view of the graph's depth at the level of a project description.

```{code-cell} ipython3
g6 = Graph()
g6.bind("fg_wd", FG_WD)
g6.bind("fg_wikibase", FG_WIKIBASE)
# define query for statements of a choosen project
qres = g6.query(
   """
SELECT ?project ?wd_labell ?ps_ ?ps_labell ?wdpq ?wdpq_labell ?pq_ ?pq_labell
WHERE {
    SERVICE <https://database.factgrid.de/sparql> {
  VALUES (?project) {(fg_wd:Q467586)}

  ?project ?p ?statement .
  ?statement ?ps ?ps_ .
  OPTIONAL { ?ps_ rdfs:label ?ps_labell.
  FILTER(lang(?ps_labell) IN ('en'))}
  ?wd fg_wikibase:claim ?p .
  ?wd fg_wikibase:statementProperty ?ps .
  OPTIONAL {?wd rdfs:label ?wd_labell.
  FILTER(lang(?wd_labell) IN ('en'))}

  OPTIONAL {
  ?statement ?pq ?pq_ .
  ?wdpq fg_wikibase:qualifier ?pq .
  ?wdpq rdfs:label ?wdpq_labell.
  FILTER(lang(?wdpq_labell) IN ('en'))
  ?pq_ rdfs:label ?pq_labell.
  FILTER(lang(?pq_labell) IN ('en'))
  }
  }
}
 ORDER BY ?wd ?ps_
   """
)
```

```{code-cell} ipython3
:tags: [table-sparkle-statement-*, data-table]

sparql_results_to_df(qres)
```

## Draw resulting graph

## TODO: network of information / agents surrounding a project

## 2.3 Querying for topics of FactGrid projects

```{code-cell} ipython3
g7 = Graph()
g7.bind("fg_wd", FG_WD)
g7.bind("fg_wdt", FG_WDT)
# define query for projects
qres = g7.query(
   """
SELECT ?topic_labell ?project_labell ?date
WHERE {
    SERVICE <https://database.factgrid.de/sparql> {
        ?project fg_wdt:P2 fg_wd:Q11295 .
	?project fg_wdt:P49 ?date .
	?project fg_wdt:P243 ?topic .
      	?project rdfs:label ?project_labell.
      	FILTER(lang(?project_labell) IN ('en'))
      	?topic rdfs:label ?topic_labell.
      	FILTER(lang(?topic_labell) IN ('en'))
   }
}
ORDER BY ?topic_labell
   """
)
```

```{code-cell} ipython3
:tags: [table-factgrid-*, data-table]

sparql_results_to_df(qres)
```

## Draw timeline

# 3.0 Collecting sub-graph of a given project

## build query

```{code-cell} ipython3
g9 = Graph()
g9.bind("fg_wd", FG_WD)
g9.bind("fg_wikibase", FG_WIKIBASE)
# define query for projects
qres = g9.query(
   """
SELECT ?s ?wd ?o
WHERE {
  SERVICE <https://database.factgrid.de/sparql> {
    {
	  BIND (fg_wd:Q467586 as ?s)
	  ?s ?p ?o.
	  ?wd fg_wikibase:directClaim ?p.
	}
	UNION
    {
	  BIND (fg_wd:Q467586 as ?o)
	  ?s ?p ?o.
	  ?wd fg_wikibase:directClaim ?p .
	}
}}
"""
)

# Due of rdflib's limitation
import rdflib
for row in qres:
    if type(row[2]) == rdflib.term.URIRef:
        g9.add(row)
```

## draw a network ?

```{code-cell} ipython3
:tags: [figure-network-3-*]

ug9 = rdflib_to_networkx_graph(g9)
Sigma(ug9)
```

## Store rdf triples

```{code-cell} ipython3
# The graph g9 is write in "my_graph.rdf" file
g9.serialize(destination="my_graph.rdf", format="xml")
```

# Creating a knowledge graph 

## Create a couple of triples with Python and `rdflib`

```{code-cell} ipython3
from rdflib import Graph, Literal, RDF, URIRef
# rdflib knows about quite a few popular namespaces, like W3C ontologies, schema.org etc.
from rdflib.namespace import FOAF , XSD

# Create a Graph
g = Graph()

# Create an RDF URI node to use as the subject for multiple triples
donna = URIRef("http://example.org/donna")

# Add triples using store's add() method.
g.add((donna, RDF.type, FOAF.Person))
g.add((donna, FOAF.nick, Literal("donna", lang="en")))
g.add((donna, FOAF.name, Literal("Donna Fales")))
g.add((donna, FOAF.mbox, URIRef("mailto:donna@example.org")))

# Add another person
ed = URIRef("http://example.org/edward")

# Add triples using store's add() method.
g.add((ed, RDF.type, FOAF.Person))
g.add((ed, FOAF.nick, Literal("ed", datatype=XSD.string)))
g.add((ed, FOAF.name, Literal("Edward Scissorhands")))
g.add((ed, FOAF.mbox, Literal("e.scissorhands@example.org", datatype=XSD.anyURI)))

# Bind the FOAF namespace to a prefix for more readable output
g.bind("foaf", FOAF)
```

## Representations of the knowledge graph

### In the `n3` serialization format

```{code-cell} ipython3
# print all the data in the Notation3 format
print(g.serialize(format='n3'))
```

### As a network graph

```{code-cell} ipython3
:tags: [figure-network-*]

from rdflib.extras.external_graph_libs import rdflib_to_networkx_graph
import networkx as nx
import matplotlib.pyplot as plt

ug = rdflib_to_networkx_graph(g)
nx.draw(ug)
```

# Session information

```{code-cell} ipython3
import session_info
session_info.show()
```

# APPENDICES - Figures with `plotly.graph_objects`

```{code-cell} ipython3
import plotly.graph_objects as go
import numpy as np

# Create figure
fig = go.Figure()

# Add traces, one for each slider step
for step in np.arange(0, 5, 0.1):
    fig.add_trace(
        go.Scatter(
            visible=False,
            line=dict(color="#00CED1", width=6),
            name="𝜈 = " + str(step),
            x=np.arange(0, 10, 0.01),
            y=np.sin(step * np.arange(0, 10, 0.01))))

# Make 10th trace visible
fig.data[10].visible = True

# Create and add slider
steps = []
for i in range(len(fig.data)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
              {"title": "Slider switched to step: " + str(i)}],  # layout attribute
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    active=10,
    currentvalue={"prefix": "Frequency: "},
    pad={"t": 50},
    steps=steps
)]

fig.update_layout(
    sliders=sliders
)

fig.show()
```
