# SO_Developer_Survey

## Funzionamento in sintesi
Il programma analizza i dati di Stack Overflow Developer Survey, 2017: in particolare calcola l'accuratezza ottenuta tramite una 10-Fold Cross Validation nel predire se il salario di un partecipante è sopra o sotto la mediana. Gli algoritmi di apprendimento usati sono Decision Tree e Random Forest, il programma è scritto in Python 3 e usa la libreria scikit-learn.

## Utilizzo del programma
Per eseguire l'esperimento, i passi sono i seguenti:

1. Procurarsi il dataset, quello usato nell'esperimento è reperibile alla pagina https://www.kaggle.com/stackoverflow/so-survey-2017. In realtà è possibile utilizzare un qualsiasi dataset che sia in formato csv e preveda una colonna per lo stipendio. È necessario inserire, all'inizio del file *exp.py* il corretto percorso al file csv, assegnando l'opportuno parametro.
2. *Opzionale*: modificare a proprio piacimento i parametri di interesse, che sono tutti definiti verso l'inizio del file *exp.py*.
3. Eseguire il file *exp.py*: alla prima esecuzione vengono generati i file intermedi, dalla seconda in poi, e se esistono già, vengono soltanto letti. Se si vuole eseguire un nuovo esperimento, è possibile cambiare i parametri relativi all'apprendimento senza che il file csv venga letto per intero nuovamente. Se si volesse anche cambiare i parametri relativi alla lettura del dataset si può fare, ma è necessario cancellare almeno uno dei file intermedi.

*Nota: i tempi di esecuzione variano sicuramente in base alla piattaforma di esecuzione, ma indicativamente sono di una decina di secondi per la preparazione dei file, e di qualche secondo per ogni test. L'avanzamento del programma viene mostrato sulla console.*

## Parametri
I parametri che regolano il funzionamento del programma sono di due tipologie: alcuni relativi alla lettura del dataset, altri relativi all'esecuzione dei test. Si riportano di seguito i parametri principali con relativa spiegazione.

Parametri relativi alla lettura del dataset:

+ fileDataset: percorso del file csv contenente il dataset;
+ separator: carattere separatore nel file csv;
+ nameColSalary: nome della colonna nella quale è contenuto lo stipendio;
+ nonAvailableValue: stringa che nel file csv rappresenta i dati mancanti;
+ acceptedNA: numero compreso tra 0 e 1 che esprime il numero di valori mancanti ammessi per colonna.

Parametri relativi all'apprendimento:

+ K: parametro K relativo alla K-Fold cross validation;
+ numTests: numero di test da eseguire;
+ maxDepth: massima profondità di ogni albero, sia per Decision Tree che per Random Forest;
+ minSamplesLeaf: numero di campioni per ogni foglia, sia per Decision Tree che per Random Forest;
+ nEstimators: numero di alberi nella foresta.