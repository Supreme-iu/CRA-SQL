# CRA-SQL
A Task-Aligned Text-to-SQL Approach with Chain-of-Thought and Retrieval-Augmented Generation

##### Data: 
Refer to the contents in this link: https://doi.org/10.5281/zenodo.17638997 in Zenodo

##### RQs: 
The main experiments mentioned in the paper (RQ1、RQ2、RQ3、RQ4).

##### Data Preprocessing：
You can find primary data processing workflow can be found here（preprocessing）

##### Core Highlights：
###### 🧠 Code-Style Chain-of-Thought (CoT)
Introduces a SQL-like code representation (SCR) as an intermediate semantic representation, decomposing complex queries into sequential and structured instructional steps to guide the model in clear and coherent logical reasoning.
###### 📚 Multi-Granularity Retrieval-Augmented Generation (RAG)
Constructs a three-layer knowledge base comprising "schema-level, instance-level, and example-level" information, dynamically incorporating relevant examples and fine-grained domain knowledge to enhance the model's generalization capability for unseen database schemas.
###### 🎯 Multi-Stage Task Alignment
Designs a framework integrating four alignment mechanisms—schema, semantic, knowledge, and output alignment—to collaboratively suppress model hallucinations (structural, semantic, and knowledge hallucinations) at each stage of SQL generation.
###### 🏆 Performance
Achieves leading performance on authoritative cross-domain benchmarks, Spider and BIRD, particularly excelling in complex query scenarios.

##### Acknowledgments：
We extend our gratitude to the Spider and BIRD teams for providing excellent benchmark datasets.
