# ml-book-code

> Official companion code repository for Data Driven: A Practical Guide to Machine Learning (《数据驱动：机器学习实战之道》)

[中文](README.md) | [English](README_EN.md)

---

## 📖 About the Book

- **Title**: Data Driven: A Practical Guide to Machine Learning (《数据驱动：机器学习实战之道》)
- **Author**: 一个处女座的程序猿 (A Virgo Programmer)
- **Publication**: Mid-2025
- **Writing**: Five years of work; a 700,000-character original manuscript and an approximately 350,000-character final manuscript

## ⭐︎ Five Features of the Book

1. **Theory and practice together**
   - Start with the concepts in the book, then use the Python examples for the matching sections to learn, run, and modify them. Chapter 1 has a conceptual guide; Chapters 2–8 have teaching scripts.
2. **Comprehensive, systematic coverage**
   - The book moves from data science lifecycles (CRISP-DM and TDSP) through visualization, data engineering, model training and inference to deployment, monitoring, and optimization.
3. **A wide range of practical examples**
   - Section-level examples cover data sampling, charts, EDA, feature processing, classification and regression, clustering, tuning, prediction APIs, A/B comparisons, and model optimization. They use included data, simulated data, or public data that must be obtained separately; see the chapter overviews for preparation details.
4. **Readable, consistent code**
   - Script names follow book sections, and Chinese step comments explain the main ideas. Different methods for one topic use labels such as `T1` and `T2` to make comparison easier.
5. **Emerging and advanced techniques**
   - Topics include model explainability, probability calibration, model ensembles, data augmentation, prediction services, monitoring, single-machine multiprocessing, and CUDA GPU computation.

---

## Companion Code at a Glance

This project accompanies 《数据驱动：机器学习实战之道》 (Data Driven: A Practical Guide to Machine Learning). It helps readers connect the book's concepts, methods, and application workflows to concrete code: how data enters an example, how its key steps are organized, and what the output can show.

The project follows the book's eight chapters and currently contains 65 Python files and eight chapter overviews. Chapter 1 focuses on concepts and has no standalone script; Chapters 2–8 contain examples organized by section. [shared/dataset.py](shared/dataset.py) is a data-loading helper imported by several chapters. The scripts are small teaching experiments or demonstrations of parts of a workflow; they are not one end-to-end project that must be run all at once.

## Reading the Code Alongside the Book

1. **Read the book section first**: Identify the problem, inputs and outputs, and key concepts.
2. **Open the chapter overview**: Use its section–file–purpose table to locate the script, suggested reading order, and resource notes.
3. **Follow the Step comments in the code**: Match `Step1, Step2, ...` to nearby code and observe how the data changes.
4. **Compare different methods within a section**: When you see `T1, T2, ...`, understand the differences before choosing what to read or run.
5. **Try a small experiment**: Change one input or parameter, predict the result, then inspect the output and return to the book to explain it. Keep data and evaluation criteria consistent when comparing model performance.

You can study chapter by chapter or go straight to the section you are reading. Follow the relevant chapter overview for each example's reading or execution order.

## 📚 Chapter Outline

### Part One: Practical Data Science Technology

1. **Chapter 1: Introduction to Data Science Technology**: Understand the value of data and lifecycle models, and compare CRISP-DM with TDSP. This chapter has a conceptual guide.
2. **Chapter 2: Data Visualization Technology**: Move from basic charts such as histograms and bar charts to combined, three-dimensional, and animated charts.
3. **Chapter 3: The Complete Data Science Workflow**: Use simulated event tracking, sampling, class imbalance handling, and initial feature screening to understand data exploration and collection. Data cleaning is covered in Chapter 5.

### Part Two: The Five Stages of the Machine Learning Workflow

4. **Chapter 4: Initial Exploratory Data Analysis (EDA)**: Explore data loading, profiling, feature-type identification, and separating features from labels. Chapter 5 covers further handling of missing values and outliers.
5. **Chapter 5: Data Engineering (Data Analysis + Data Processing)**: Work through cleaning, analysis, feature construction, scaling, encoding, vectorization, feature selection, and export.
6. **Chapter 6: Model Training, Evaluation, and Inference**: See classification, regression, and clustering examples alongside model evaluation, tuning, error analysis, export, and inference.
7. **Chapter 7: Model Release, Deployment, and Monitoring**: Use simulated A/B comparisons, a local prediction API, and monitoring examples to understand release, deployment, and monitoring.
8. **Chapter 8: Full-Project Analysis, Review, and Optimization**: Compare regularization, data augmentation, model ensembles, and code efficiency, then read the conceptual material on system optimization.

## Chapter Guide

The table follows the order of the book. Click a chapter title to open its overview and find individual sections and scripts.

| Book chapter | Companion-code focus | Code files |
| --- | --- | --- |
| [Chapter 1: Introduction to Data Science Technology](一、数据科学技术的简介/CHAPTER_OVERVIEW.md) | Data science concepts, CRISP-DM, and TDSP | Chapter guide |
| [Chapter 2: Data Visualization Technology](二、数据可视化技术/CHAPTER_OVERVIEW.md) | Basic, combined, three-dimensional, and animated charts | 5 Python |
| [Chapter 3: The Complete Data Science Workflow](三、数据科学任务完整流程/CHAPTER_OVERVIEW.md) | Data channels, sampling, imbalance handling, and initial feature screening | 5 Python |
| [Chapter 4: Initial Exploratory Data Analysis (EDA)](四、初步探索性数据分析%28EDA%29/CHAPTER_OVERVIEW.md) | Data loading, profiling, feature types, and separating labels | 4 Python |
| [Chapter 5: Data Engineering (Data Analysis + Data Processing)](五、数据工程%28数据分析%2B数据处理%29/CHAPTER_OVERVIEW.md) | Cleaning, feature construction, transformation, selection, and export | 15 Python |
| [Chapter 6: Model Training, Evaluation, and Inference](六、模型训练、评估与推理/CHAPTER_OVERVIEW.md) | Training, metrics, tuning, error analysis, and inference | 13 Python |
| [Chapter 7: Model Release, Deployment, and Monitoring](七、模型发布、部署与监控/CHAPTER_OVERVIEW.md) | A/B comparisons, prediction APIs, and monitoring | 7 Python |
| [Chapter 8: Full-Project Analysis, Review, and Optimization](八、模型项目整体性分析、反思与优化/CHAPTER_OVERVIEW.md) | Model fit and improvements to data, algorithms, systems, and code | 15 Python |

Chapter 1 focuses on concepts and lifecycle models. It has a chapter overview but no standalone code. The counts for other chapters include only Python files in their chapter directories; the cross-chapter helper is in `shared/`.

## File and Comment Conventions

- **Section numbers**: The number at the start of a filename identifies the matching book section. Refer to the book, file header, and chapter overview for its learning objective.
- **`Step1, Step2, Step3`**: These mark the example's main processing steps, with comments placed beside the relevant code.
- **`T1, T2`**: These identify different methods or approaches for one topic. Some are in separate files; others appear in the same file.
- **Language and terminology**: Explanatory code comments and chapter overviews are primarily in Chinese. Terms such as API and EDA, interface fields, function names, and commands keep their original spelling. The English README uses the same file paths.
- **Resource paths**: Repository resources use paths relative to the project root; run scripts from the directory containing this README. Scripts importing `shared.dataset` also need that directory on Python's module search path.

| Example type | How to read and use it |
| --- | --- |
| Small concept or algorithm example | Focus on inputs, processing steps, and outputs; compare the methods in the book before running it as needed. |
| Data processing and visualization example | Check the data and what each chart shows; consult the chapter overview for external data, graphical windows, or fonts. |
| Model, service, and monitoring example | Prepare the required dependencies, data, and local services first; simulated results do not represent live business outcomes. |
| Cross-chapter helper | [shared/dataset.py](shared/dataset.py) is imported by scripts in several chapters and usually does not need to be run on its own. |

## Start with Two Small Examples

Prepare a Python 3 environment and open a terminal in the project root. These two examples use only the Python standard library, with no data downloads or third-party dependencies.

### Compare Simple Random and Stratified Sampling

Corresponding book section: 3.2.4.2, common data sampling methods.

~~~bash
python3 "三、数据科学任务完整流程/3.2、数据认知与数据收集/3.2.4、数据采样/3.2.4.2_random_and_stratified_sampling.py"
~~~

Compare the class proportions produced by the two methods. Then change the sample size or population proportions and consider what stratification does, and does not, guarantee.

### Observe When a SQL Default Takes Effect

Corresponding book section: 5.1.4.2, practical code for analyzing and handling special values.

~~~bash
python3 "五、数据工程(数据分析+数据处理)/5.1、数据清洗/5.1.4_sql_default_values.py"
~~~

Compare omitting a field with explicitly inserting `0` or `NULL` to see how `DEFAULT` differs from an explicit value.

If your environment starts Python 3 with `python` or Windows' `py -3`, substitute that for `python3`. Keep the quotation marks when copying paths.

## Prepare Environments and Resources for Other Examples

### Choose Dependencies for Each Example

[conda-spec-win64.txt](conda-spec-win64.txt) is a **Conda environment export for Windows `win-64`**, including build identifiers. It cannot be used directly with `pip install -r`. On another system, use its libraries and versions as a reference for a compatible environment, then check the selected script's imports and chapter overview for the components needed. Not every example needs every package in the list.

| Example area | Common components; select those needed by the script |
| --- | --- |
| Visualization and EDA | pandas, Matplotlib, Seaborn, and libraries required by the particular chart or report |
| Data engineering and model training | NumPy, scikit-learn, and libraries such as imbalanced-learn or LightGBM used by the selected example |
| Deep learning and performance analysis | PyTorch or TensorFlow; CUDA examples also need a compatible GPU environment |
| Local prediction service | Flask, LightGBM, and requests for the request examples |

### Data and Local Services

- Some examples fetch public data from OpenML, MNIST, or UCI at runtime and may need network access on first use. Others read included CSV files or construct simulated data. See the matching chapter overview for resources and run conditions.
- Chapter 7 prediction requests require the local API service to be running first. Its A/B comparisons and some monitoring data are teaching simulations, not real business experiments.
- Examples importing [shared/dataset.py](shared/dataset.py) need the repository root on Python's module search path. From the root on macOS/Linux, for example, use `PYTHONPATH=. python3 "script path"`. In Windows Command Prompt, first run `set PYTHONPATH=.`.
- Images, webpages, logs, and data produced at runtime can be regenerated as needed. Distinguish input resources from script outputs when reading an example.

## Interpreting Example Results

- **Relate the scope to the book's objective**: One script usually shows one concept or part of a workflow. Identify what it explains before extending it.
- **Distinguish simulated from real data**: Simulated event tracking, A/B metrics, and synthetic samples show method structure; they do not establish real business effectiveness.
- **Keep comparison conditions consistent**: Hold data splits and evaluation criteria steady when comparing sampling, models, or optimization methods. Timing also depends on hardware and environment.
- **Interpret outputs in their resource context**: Public dataset versions, randomness, GPUs, and local service state can affect results. Use actual outputs and the book's definitions as the basis for interpretation.

## Sources and Feedback

The book is the reference for chapter content and learning order. For examples involving public datasets or third-party libraries, follow the terms of their respective sources.

When reporting a problem, include the book section, relative code path, Python and key dependency versions, minimal reproduction steps, and expected and actual results. This helps identify the issue in relation to the example's learning objective.
