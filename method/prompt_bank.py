column_meaning_prompt = """def convert_schema_to_comprehensive_description(db_id, table_name, column_name, column_type, column_description = None, value_description = None):
    # step1: The interpretation of a column name is contingent upon its relational association with the table name. Thus, the first generated sentence should explain the column meaning within the context of table_name
    # step2: output overall column description according to step1
    assert len(overall_description) <= 100
    return overall_description

overall_description = convert_schema_to_comprehensive_description({input_paras})

print(overall_description)

#Output: """



dummy_sql_prompt = """# Task: Convert natural language questions to accurate SQL queries for SQLite

# Database Schema Documentation
# Key: table name
# Value: dictionary with column information (full name, description, value explanation, examples)
database_schema = {database_schema}

# Primary Key Relationships
# Key: table name
# Value: list of primary key columns
primary_keys = {primary_key_dic}

# Foreign Key Relationships
# Key: source column
# Value: referenced target column
foreign_keys = {foreign_key_dic}

# User Question to Answer
question = "{question_prompt}"

# Supporting Evidence Context
evidence = "{evidence}"

def question_to_SQL(question):
    \"\"\"Generate SQLite-compliant SQL query following these strict rules:

    1. COLUMN SELECTION:
    - Select ONLY columns explicitly requested in the question
    - Use EXACT column names from database_schema (case-sensitive)
    - Never include columns not mentioned in the question

    2. SYNTAX REQUIREMENTS:
    - Use only SQLite-supported functions (no YEAR(), DATEDIFF(), etc.)
    - For year extraction: use strftime('%Y', date_column)
    - For date differences: use julianday() arithmetic
    - Properly quote identifiers when needed

    3. TABLE RELATIONSHIPS:
    - Explicitly specify JOIN conditions using primary/foreign keys
    - Verify join paths exist in foreign_keys mapping
    - Include all necessary tables (no missing FROM clauses)

    4. QUERY STRUCTURE:
    - Single statement only (no multiple queries)
    - Validate GROUP BY/HAVING usage:
      * HAVING only with aggregate functions
      * GROUP BY must include all non-aggregate SELECT columns
    - Use proper subquery syntax when needed

    5. VALIDATION CHECKS:
    - Cross-reference every column/table with database_schema
    - Verify all functions are SQLite-compatible
    - Check JOIN conditions match foreign_key relationships
    - Ensure proper WHERE clause syntax

    6. EVIDENCE INTEGRATION:
    - Incorporate relevant evidence into query logic
    - Use evidence to refine WHERE conditions when applicable

    Failure to follow these rules will result in SQL errors.
    \"\"\"

    # Generate SQL query here following all above requirements
    ### STRICT OUTPUT FORMAT REQUIREMENTS:
    # Generate executable SQL (must start with SELECT):
    # You MUST wrap the generated SQL in markdown code blocks like this:
    ```sql"""


sr_examples = """#SR is a piece of pandas-like code. Learn to generate SR based on the question and the schema. Later, the SR will be converted to SQL. 
#SR ignore 'join' action. Do not generate 'join' action.
#In the generated SR, only select the thing that request in the question. Do not select any non-requested stuff. 
#The filter condition in the 'where' function doesn't directly match the text in the question. To find the correct value for the 'where' function, you need to reference the example values or all possible values in column description.

question = "How many movies directed by Francis Ford Coppola have a popularity of more than 1,000? Please also show the critic of these movies."
schema = [movies.movie_title, ratings.critic, movies.director_name, movies.movie_popularity, ratings.movie_id, movies.movie_id']
evidence = "Francis Ford Coppola refers to director_name; popularity of more than 1,000 refers to movie_popularity >1000" 
SR = "df1 = df.where(element = movies.director_name, filter = 'Francis Ford Coppola')
    df2 = df1.where(element = movies.movie_popularity, filter = '> 1000')
    res = df2.select(movies.movie_title, ratings.critic)"
    
question = "For all the movies which were produced by cruel and unusual films, which one has the most popularity?"
schema = [movie.title, `production company`.company_id, movie_company.company_id, movie_company.movie_id, movie.movie_id, `production company`.company_name, movie.popularity]
evidence = "produced by cruel and unusual films refers to company_name = 'Cruel and Unusual Films'; most popularity refers to max(popularity)"
SR = "df1 = df.where(element = `production company`.company_name, filter = 'Cruel and Unusual Films')
    df2 = df1.orderby(by = movie.popularity, desc).limit(1)
    res = df2.select(movie.title)"
    
question = "Among the professors who have more than 3 research assistants, how many of them are male?"
schema = [prof.gender, RA.student_id, RA.prof_id, prof.prof_id]
evidence = "research assistant refers to the student who serves for research where the abbreviation is RA; more than 3 research assistant refers to COUNT(student_id) > 3;"
SR = "df1 = df.groupby(prof.prof_id).having(element = count(RA.student_id), filter = '> 3')
    df2 = df1.where(element = 'prof.gender', filter = 'Male')
    res = df2.count()"

question = "What is the first name of clients who have the highest priority?"
schema = [client.first, client.client_id, callcenterlogs.`rand client`,callcenterlogs.priority]
evidence = "first name refers to first; highest priority refers to priority = 2"
SR = "df1 = df.where(element = callcenterlogs.priority, filter = max(callcenterlogs.priority))
    res = df1.select(client.first)"

question = "What percentage of businesses in the northwest US have forecasted annual sales of above 300,000?"
schema = [SalesPerson.SalesQuota, SalesPerson.BusinessEntityID, SalesPerson.TerritoryID, SalesTerritory.TerritoryID, SalesTerritory.CountryRegionCode, SalesTerritory.Name]
evidence = "northwest refers to Name = 'Northwest'; US refers to CountryRegionCode = 'US'; forecasted annual sales of above 300,000 refers to SalesQuota >300000; Percentage = Divide(Count(TerritoryID(SalesQuota >300000)),Count(TerritoryID))*100"
SR = "df1 = df.where(element = SalesTerritory.CountryRegionCode, filter = 'US').where(element = SalesTerritory.Name, filter = 'Northwest')
    df2 = df1.where(element = SalesPerson.SalesQuota, filter = '> 300000')
    res = df.select(cast(df2.count(), real) * 100 / df1.count())"

question = "How many more followers in percentage are there for the repository used by solution ID 18 than solution ID19?"
schema = [Repo.`Forks Number`, Solution.Id, Repo.Id, Solution.RepoId]
evidence = "followers refers to Forks; percentage = divide(SUBTRACT(Forks(Solution.ID = 18), Forks(Solution.ID = 19)), Forks(Solution.ID = 19))*100%"
SR = "df1 = df.where(element = Solution.Id, filter = 18))
    tmp_res1 = df1.select(Repo.`Forks Number`).sum()
    
    df1 = df.where(element = Solution.Id, filter = 19))
    tmp_res2 = df1.select(Repo.Forks).sum()"
    
    res = df.select(cast((tmp_res1 - tmp_res2),real) * 100 / tmp_res2)"
    
question = "What is the difference between the number of children's films and action films?"
schema = [category.name, film_category.category_id, category.category_id]
evidence = ""
SR = "df1 = df.where(element = category.name, filter = 'ChildrenFilm')
    df2 = df.where(element = category.name, filter = 'ActionFilm')
    res = df.select(df1.count() - df2.count())"
"""


generate_sr = """{sr_example}
column_description = {column_description}
question = {question}
schema = {schema}
evidence = "{evidence}"
SR =
"""

sr2sql = """
# Understand the pandas-like SR first. Then convert the SR into executable SQL based on:
# - The current question
# - Retrieved similar examples
# - Schema description
# - Foreign key relationships
# Notice the order of the action in SR may not be the same as the executable SQL. Make sure the generated SQL is executable and can answer the question accurately according to the schema.
# Only select the thing that the question required. Do not select any non-requested stuff.
# You may need to look back at the column_description and schema to get the correct value used in the final SQL.

# Critical SQLite-Specific Rules:
1. **Use SQLite syntax** (e.g., `QUALIFY` and `YEAR()` are NOT supported).
2. **Never assume column names exist** — strictly use only columns listed in the schema.
3. **Avoid unsupported functions** like `DATEDIFF`, `IF`, or `YEAR()`. Use alternatives (e.g., `strftime` for dates).
4. **Pagination must use `LIMIT`** (do not use `TOP` or `FETCH`).

# Retrieved Similar Examples:
{examples}

# Current Task Details:
question = {question}
schema = {schema}
evidence = "{evidence}"
SR = "{SR}"
# "'df', 'df1', and 'df2' are not actual tables and should not appear in the generated SQL."

# Schema Description:
column_description = {column_description}

# Foreign Key Relationships:
foreign_keys = {foreign_key_dic}

# Available SQL Keywords:
from CLAUSE_KEYWORDS import select, from, where, group by, order by, union, limit, having, distinct, as, between, like, all, on, partition by
from JOIN_KEYWORDS import inner join, left join
from WHERE_OPERATIONS import is, not, null, none, in, =, >, <, >=, <=, !=, <>
from DATE_OPERATIONS import now, curdate, strftime
from UNIT_OPERATIONS import -, +, *, /
from COND_OPERATIONS import and, or, case, iif
from SQL_OPERATIONS import avg, count, max, min, round, abs, sum, length, cast, substr, instr
from ORDER_OPERATIONS import desc, asc

### STRICT OUTPUT FORMAT REQUIREMENTS:
# Generate executable SQL (must start with SELECT):
# You MUST wrap the generated SQL in markdown code blocks like this:
```sql
"""
