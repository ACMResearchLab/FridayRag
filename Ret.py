import ast
import argparse
from openai import OpenAI
import numpy as np
from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.preprocessing import normalize
from scipy.spatial.distance import cosine
# from sentence_transformers import SentenceTransformer

client = OpenAI(api_key='key')



#file parser
def parse_functions(code):
    tree = ast.parse(code)

    functions = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            functions.append(ast.unparse(node))
    return functions



#code->english
def translate(funcs):
    translated_functions = []
    
    promptsys = f"Summarize the functions into english within 20 tokens."
    msgs = [{"role": "system", "content": promptsys}]
    for func in funcs:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Generate a short english summary of the following function within 20 tokens:"},
                {"role": "user", "content": func},
            ],
            temperature=0,
            max_tokens=25
        )
        translation = response.choices[0].message.content
       
        translated_functions.append(translation)
    return translated_functions



# #normalize function, not needed?
# def normalize_l2(x):
#     x = np.array(x)
#     if x.ndim == 1:
#         norm = np.linalg.norm(x)
#         if norm == 0:
#             return x
#         return x / norm
#     else:
#         norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
#         return np.where(norm == 0, x, x / norm)


#embedder
def embed(nlin, mod):
    if (mod == 'openai'):
        response = client.embeddings.create(
            input=nlin,
            model="text-embedding-3-small",
        )
        x = np.array(response.data[0].embedding)
     
    else:
        model = SentenceTransformer(mod)
        x=np.array((model.encode(nlin))[0])
        
    

    return x



def rank_functions(query, translated_functions, model, k):
    query_embedding = embed([query], model)
    function_embeddings = [embed([func], model) for func in translated_functions]
    ranks = {}

    for i, func_embedding in enumerate(function_embeddings):
        similarity = cosine(query_embedding, func_embedding)
        ranks[i] = similarity


    sorted_ranks = {k: v for k, v in sorted(ranks.items(), key=lambda item: item[1])}

    top_k_ranks = dict(list(sorted_ranks.items())[:k])

    return top_k_ranks



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Name of the Python file containing functions (context)")
    parser.add_argument("query", help="Query to rank functions against (query)")
    parser.add_argument("model", help="Name of the SentenceTransformer model to use, openai if you want to use there model for embeddings")
    parser.add_argument("k", type=int, help = "How many matches should be returned, ie top k matches.")
    args = parser.parse_args()

    with open(args.file) as file:
        code = file.read()

    funcs = parse_functions(code)
    eng_funcs = translate(funcs)
    ranked_functions = rank_functions(args.query, eng_funcs, args.model, args.k)

    #return code
    # for i, (index, similarity) in enumerate(ranked_functions.items(), 1):
    #     print(f"Rank {i}: Similarity: {similarity}, Function: {funcs[index]}")


    #return english
    for i, (index, similarity) in enumerate(ranked_functions.items(), 1):
        print(f"Rank {i}: Similarity: {similarity}, English Summary: {eng_funcs[index]}")