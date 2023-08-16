<!-----

Yay, no errors, warnings, or alerts!

Conversion time: 0.467 seconds.


Using this Markdown file:

1. Paste this output into your source file.
2. See the notes and action items below regarding this conversion run.
3. Check the rendered output (headings, lists, code blocks, tables) for proper
   formatting and use a linkchecker before you publish this page.

Conversion notes:

* Docs to Markdown version 1.0β34
* Wed Aug 16 2023 04:09:05 GMT-0700 (PDT)
* Source doc: Attentional Search

WARNING:
You have 3 H1 headings. You may want to use the "H1 -> H2" option to demote all headings by one level.

----->


<p style="color: red; font-weight: bold">>>>>>  gd2md-html alert:  ERRORs: 0; WARNINGs: 1; ALERTS: 0.</p>
<ul style="color: red; font-weight: bold"><li>See top comment block for details on ERRORs and WARNINGs. <li>In the converted Markdown or HTML, search for inline alerts that start with >>>>>  gd2md-html alert:  for specific instances that need correction.</ul>

<p style="color: red; font-weight: bold">Links to alert messages:</p>
<p style="color: red; font-weight: bold">>>>>> PLEASE check and correct alert issues and delete this message and the inline alerts.<hr></p>



# Attentional Search

Project website: [http://attentionalsearch.media.mit.edu](http://attentionalsearch.media.mit.edu/)

Github repository: [https://github.com/ptimons44/attentional-search](https://github.com/ptimons44/attentional-search)

Attention Bert model: [https://huggingface.co/ptimons44/attention-bert](https://huggingface.co/ptimons44/attention-bert)


# Instructions



1. Think of a search query appropriate for Attentional Search, as detailed below in the background section.
2. Type your search query into the search bar, choose the number of search queries by using either the slider or the input field, and press submit. Note that there may not be an immediate update to the page, and you will have to wait around a minute for the initial update.
3. The first update will be an annotated response to your query from an external Large Language Model (LLM). The annotated response is highlighted in varying intensities. Darker highlighting indicates higher aggregate attention paid to a particular word. 
4. Once the LLM response is displayed, the server gets to work extracting search queries and scraping the web based on the attention values of the LLM response. Please allow a few minutes for this step, as there is a massive amount of data being processed.
5. Once the above task completes, a 3D plot of the most relevant sentences found from the web should render underneath the annotated LLM response. Each plotted sentence is one of the top k most relevant sentences found in the search process. Sentences closer to each other in the 3D space have more similar meanings than sentences that are further apart. It is essentially a human-readable visualization of the sentences plotted in [embedding space](https://en.wikipedia.org/wiki/Word_embedding). You can interact with this plot by zooming and mousing around, or clicking on sentences to expand their content and metadata. 
6. Sentences are color coded by search query, and you can filter which search queries are plotted by unchecking search queries from the checklist below the sentence plot.
7. That’s it—Happy researching!


# Background

With the advent of highly capable Large Language Models, It seems as though more and more search queries will be directed to LLMs instead of traditional search engines. Google supposedly announced an internal code red because it recognizes the power of emerging technologies such as OpenAI’s ChatGPT, and is worried that its search moat may be thinner than presumed.

Regardless of what the future of search looks like for most end users, it will remain critical that users can access up to date, accurate information. This is a notable drawback of Large Language Models, as they learn to respond to queries based on training data that could be years old. Its responses are always based on old data since there is no native real-time information recall. In addition to the temporal deficiency, Large Language Models are black boxes to the end user who can not see the data that it was trained on. Luckily, this data largely lives all over the internet, and is indexable by search engines such as Google’s.

Additionally, few-shot prompting has shown to be very effective in improving LLM response accuracy. Thus a method to obtain updated information from the web and reprompt the LLM may serve as a memory robust to temporal changes in data.

During my 8 week internship with the UniSA Wearable Computer Lab, I identified these problems and engineered a product that serves as my initial contribution to the problems identified.


# Project Description

Attentional Search is a search engine that leverages the power of existing search engines and Large Language Models to elevate one’s search power. It applies novel insights with traditional Natural Language Processing techniques to tackle the problem of search. 

The Attentional Search approach is much more comprehensive than typical search, and is aimed at search queries that are more complex in nature and may require real-time data, which is currently impossible in LLMs. This category of search, rather than quick searches, is its primary target due to the computational complexity of its work. 


## Pipeline

A user enters a search query such as “What is computer attention, and how could it be useful for a search engine?” or “What are the most dangerous animals in Australia” or “Is the news about Trump in the past 24 hours true?”

 \
The query is then sent to a LLM (ChatGPT 3.5), where we get its response. Given the LLM response, we extract k search queries to then search in a traditional search engine (Google). We extract the search queries based on self-attention values. We choose the k words with highest incoming attention, and group them with words that they attend highly to. 

We get u urls per search query, and scrape the content from all the collected urls. We then split the content into sentences, and embed each sentence using a contextual model (OpenAI embeddings). We then get the top n sentences in cosine similarity to the embedding of the LLMs response. 

Finally, we aim to represent the sentences in a human-readable modified embedding space. To achieve this, we use the t-SNE technique to reduce sentence embedding to 3-dimensional space.

Sentences appear as nodes in the 3D graph that a user can mouse over and choose to further explore if desired by clicking, as with a normal search engine. This interface allows for greater interactivity and rapid examination of hundreds of web pages in a few seconds.  


## Technical Accomplishments



* Novel search query generation idea
* Created custom transformer model and hosted on Hugging Face
    * [https://huggingface.co/ptimons44/attention-bert](https://huggingface.co/ptimons44/attention-bert)
* Created web application ready to scale up with deployment needs by utilizing web framework and task queues
* User Interface and presentation of search results distinguished from typical search engine


## Design choices 

We offload all model inference to external servers. The attention score calculation is done on a Hugging Face server accessed through the Inference Client Endpoint for a custom implementation of the Bert base model. We use OpenAI embeddings (using OpenAI API) when searching for the most relevant sentences.

We use a celery task queue to parallelize the sentence retrieval process. This enables scalability as the platform grows and needs to accommodate higher traffic throughput. 

We display sentences in a 3-dimensional reduced embedding space where spatial locality entails propinquity in embedding space. Determining how much to compress sentence embeddings posits the tradeoff of maintaining nuanced semantic differences between sentences and clarity of information presentation. We choose to prioritize ease of understanding over accuracy since the primary objective of a search engine is to present information in a digestible manner. Since humans have a difficult time intuiting higher dimensional objects, we chose to stick with the 3-dimensional representation.

The web application is built with the Python framework Dash, and its visuals are created with its complementary library Plotly. Although a higher degree of customization is attainable with other web technologies such as React and JavaScript, as a Data Scientist concerned primarily with the collection and organization of data, I forwent the extra step to go above and beyond in presenting the data to settle for something adequate, although I readily admit that working with a framework as simple as Dash is limiting.


## Next Steps and Future Work

This website is merely a light demonstration of the intuitive power of attention applied to the process of research. Although with adequate resources it could be scaled up to accommodate a large user base, that is not the intention, as the Attentional Search process is resource intensive. It is primarily a demonstration of the power of extracting human insights from computer attention.

Although Neural Networks often learn representations that differ fundamentally from human representations, the success of this project has made clear that computer attention can shed intuitive insights about human attention. As model architectures continue to evolve, they approach a reasoning process that more and more mimics that of an authentic human. Thus I posit the problem of indexing information by attention values. This is an unfathomably broad focus, as attention is one of the fundamental mechanisms by which humans interact with and understand information. Nonetheless, I embrace this exciting journey and look forward to investigating more scalable methods to index and represent information by attention.
