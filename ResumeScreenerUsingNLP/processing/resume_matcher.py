from preprocessing import docx_processing  as doc, textract_processing as txt
from text_processing import tf_idf_cosine_similarity as tf_idf,doc2vec_comparison as d2v
from text_processing import cv_cosine_similarity as cv
import os



def process_files(req_document,resume_docs):
    req_doc_text = txt.get_content_as_string(req_document)
    # print('The start' * 5)
    resume_doc_text = []
    for doct in resume_docs:
        resume_doc_text.append(txt.get_content_as_string(doct))

    cos_sim_list = tf_idf.get_tf_idf_cosine_similarity(req_doc_text,resume_doc_text)
    final_doc_rating_list = []
    zipped_docs = zip(cos_sim_list,resume_docs)
    sorted_doc_list = sorted(zipped_docs, key = lambda x: x[0], reverse=True)
    for element in sorted_doc_list:
        doc_rating_list = []
        doc_rating_list.append(os.path.basename(element[1]))
        doc_rating_list.append("{:.0%}".format(element[0]))
        final_doc_rating_list.append(doc_rating_list)
    return final_doc_rating_list
   






