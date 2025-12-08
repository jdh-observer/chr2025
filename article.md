---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

---
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---


# Introduction and Workshop Scope : "RDF Dump"

The workshop offers a playground notebook to create or to explore RDF dumps. If you are unfamiliar with linked data, we recommend reading the [corresponding lesson of "Programming Historian"](https://doi.org/10.46430/phen0068).

This workshop offers a brief practical introduction of how to browse into RDF dumps or how to build RDF graphs for database exploration and open research purposes. Basic know how to query or build LOD data, basic analysis for exploration, here within python language paradigm, and editorial issues should be in the hands of trainees.
This workshop is split in five parts, plus further reading :
1. A first part introduces python packages to be used within the general framework of jupyter notebooks. Some ressource gives further tutorial to work on your local machine.
2. A second part gives python recepies to browse into wikidata and Factgrid wikibase dump to collect general and particular data from the graph. 
3. A third part help in building RDF graph with python `rdflib` packages and fonctions.
4. A fourth part uses SPARQL queries results to display them on maps or timelines, leading to first results as examples. 
5. A last part aims at producing a scientific publication within the Journal of Digital History editorial technical stack.

## From data collection to scientific publication

![Ressources to install Jupyter Notebook framework on your local machine](https://mise-en-pratique-5e5223.gricad-pages.univ-grenoble-alpes.fr/jupyter.html)

```{iframe} https://blog.factgrid.de/
:align: center
:label: factgrid-blog
Factgrid blog [https://blog.factgrid.de/](https://blog.factgrid.de/)
```
```{figure} ./img/jdh.png
:align: center
:label: jdh
Narrative, hermeneutical and code layers of the Journal of Digital History, image captured from the JdH article [https://doi.org/10.1515/JDH-2023-0018](https://doi.org/10.1515/JDH-2023-0018)
```

# Python libraries to perform SPARQL queries

## Python packages

- `rdflib` : working with RDF data
- `networkx` : managing graphs
- `matplotlib` : plotting
- `plotly` : plotting as wel

## A "reproducible" process ?

- `session_info` : retrieve software version numbers

## Install packages

In your terminal, you can install these packages with `pip` :

```{code} shell
pip install session_info rdflib networkx matplotlib plotly
```

# Creating a knowledge graph 

## Create a couple of triples with Python and `rdflib`

```{code-cell} python
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

```{code-cell} python
# print all the data in the Notation3 format
print(g.serialize(format='n3'))
```

### As a network graph

```{code-cell} python
from rdflib.extras.external_graph_libs import rdflib_to_networkx_graph
import networkx as nx
import matplotlib.pyplot as plt

ug = rdflib_to_networkx_graph(g)
nx.draw(ug)
```

# Figures with `plotly.graph_objects`

```{code-cell} python
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
# Before browsing FactGrid wikibase dump

## Namespaces and prefixes

Many Widely used namespaces are predefined in `rdflib`, e.g. `FOAF`, `DublinCore`, `Schema`, or `XSD`.

```{code-cell} python
for prefix,ns in g.namespaces():
   print(prefix,ns)
```

However, we see no `wikibase`, nor `wikidata` or `dbpedia` namespaces. These have to be declared and bound to specific prefixes.

## Querying Wikidata

 ```{code-cell} python
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
```{code-cell} python
# print results nicely
for row in qres:
   print("wikidata identifier Q12418 <has label> %s" %row)
```

# Querying FactGrid

## 1.0 - Collecting keywords

```{code-cell} python
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
SELECT DISTINCT ?p (COALESCE(?p_labell,'') AS
?p_label)
WHERE {
    SERVICE <https://database.factgrid.de/sparql> {
?p fg_wdt:P1132 fg_wd:Q960698.
OPTIONAL {
?p rdfs:label ?p_labell.
FILTER(lang(?p_labell) IN
('en'))
}
}
}
ORDER BY ?p
   """
)
```

```{code-cell} python
# print results nicely
for row in qres:
   print(row.asdict()['p_label'])
```

## TODO : classes of keywords ?

## 2.0 - Collecting Factgrid projects names etc.

```{code-cell} python
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

```{code-cell} python
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

```{code-cell} python
sparql_results_to_df(qres)
```
## 2.0 TODO : map of location of projects (proportional circles ? heat map ?)

## 2.1 Querying the number of statements per projects

```{code-cell} python
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
```{code-cell} python
sparql_results_to_df(qres)
```
## Build query for project as domain and as range

```{code-cell} python
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

```{code-cell} python
from rdflib.extras.external_graph_libs import rdflib_to_networkx_graph
import networkx as nx
import matplotlib.pyplot as plt

ug8 = rdflib_to_networkx_graph(g8)
nx.draw(ug8)
```



## 2.2 Querying for the list of statements for a given project

A view of the graph's depth at the level of a project description.

```{code-cell} python
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

```{code-cell} python
sparql_results_to_df(qres)
```

## Draw resulting graph

## TODO : network of information / agents surrounding a project

## 2.3 Querying for topics of Factgrid projects

```{code-cell} python
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
```{code-cell} python
sparql_results_to_df(qres)
```

## Draw timeline

# 3.0 Collecting sub-graph of a given project

## build query

```{code-cell} python
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
for row in qres:
    g9.add(row)

print(g9.serialize(format='n3'))
```
## draw a network ?

## Store rdf triples

# Session information

```{code-cell} python
import session_info
session_info.show()
```
