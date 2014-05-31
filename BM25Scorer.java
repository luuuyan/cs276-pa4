package edu.stanford.cs276;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class BM25Scorer extends AScorer
{
	Map<Query,Map<String, Document>> queryDict;
	
	public BM25Scorer(Map<String,Double> idfs,Map<Query,Map<String, Document>> queryDict)
	{
		super(idfs);
		this.queryDict = queryDict;
		this.calcAverageLengths();
	}

	
	///////////////weights///////////////////////////
    double urlweight = 3;
    double titleweight  = 5;
    double bodyweight = 0.5;
    double headerweight = 1;
    double anchorweight = 5;
    
    ///////bm25 specific weights///////////////
    double burl=0.75;
    double btitle=0.75;
    double bheader=0.75;
    double bbody=0.75;
    double banchor=0.75;

    double k1=2;
    double pageRankLambda=3;
    double pageRankLambdaPrime=2;
    //////////////////////////////////////////
    
    ////////////bm25 data structures--feel free to modify ////////
    
    Map<Document,Map<String,Double>> lengths;
    Map<String,Double> avgLengths;
    Map<Document,Double> pagerankScores;
    
    //////////////////////////////////////////
    
    //sets up average lengths for bm25, also handles pagerank
    public void calcAverageLengths()
    {
    	lengths = new HashMap<Document,Map<String,Double>>();
    	avgLengths = new HashMap<String,Double>();
    	pagerankScores = new HashMap<Document,Double>();
    	
		/*
		 * @//TODO : Your code here
		 */
    	double sumUrlLength = 0.0;
    	double sumTitleLength = 0.0;
    	double sumBodyLength = 0.0;
    	double sumHeaderLength = 0.0;
    	double sumAnchorLength = 0.0;
    	
    	int totalDocCount = 0;
    	
    	for (Query query: queryDict.keySet()){
    		for (String url : queryDict.get(query).keySet()){
    			totalDocCount ++;
    			
    			Document doc = queryDict.get(query).get(url);
				double urlLength = getGenLength(doc.url);
				double titleLength = getGenLength(doc.title);
				double bodyLength = getBodyLength(doc);
				double headerLength = getHeaderLength(doc);
				double anchorLength = getAnchorLength(doc);
				
				sumUrlLength += urlLength;
				sumTitleLength += titleLength;
				sumBodyLength += bodyLength;
				sumHeaderLength += headerLength;
				sumAnchorLength += anchorLength;
				
				// store field length for document
				lengths.put(doc, new HashMap<String, Double>());
				lengths.get(doc).put("url", urlLength);
				lengths.get(doc).put("title", titleLength);
				lengths.get(doc).put("body", bodyLength);
				lengths.get(doc).put("header", headerLength);
				lengths.get(doc).put("anchor", anchorLength);
				
				// smooth pagerank
				pagerankScores.put(doc, Math.log10(pageRankLambdaPrime + doc.page_rank));
    		}
    	}
    	
    	
    	//normalize avgLengths

    	avgLengths.put("url", sumUrlLength / totalDocCount);
		avgLengths.put("title", sumTitleLength / totalDocCount);
		avgLengths.put("body", sumBodyLength / totalDocCount);
		avgLengths.put("header", sumHeaderLength / totalDocCount);
		avgLengths.put("anchor", sumAnchorLength / totalDocCount);

//		System.out.println("avgLengths: " + avgLengths);

    }
    
    ///////////////////// get lengths for fields /////////////////
    
    // get length for url, title
    public double getGenLength(String s){
    	return (double) s.split("\\W+").length;
    }
    
    // get body length
    public double getBodyLength(Document d){
    	return (double) d.body_length;
    }
    
    // get header length
    public double getHeaderLength(Document d){
    	double sumOfHeaderLength = 0.0;
    	if (d.headers != null){
    		for (String header : d.headers){
    			sumOfHeaderLength += header.split("\\W+").length;
    		}
    	}
    	return sumOfHeaderLength;
    }
    
    // get anchor length
    public double getAnchorLength(Document d){
    	double sumOfAnchorLength = 0.0;
    	if (d.anchors != null){
    		for (String anchor : d.anchors.keySet()){
    			int anchorCount = d.anchors.get(anchor);
    			sumOfAnchorLength += (anchor.split("\\W+").length * anchorCount);
    		}
    	}
    	return sumOfAnchorLength;
    }
    
    
    ////////////////////////////////////
    
    
	public double getNetScore(Map<String,Map<String, Double>> tfs, Query q, Map<String,Double> tfQuery,Document d)
	{
		double score = 0.0;
		
		/*
		 * @//TODO : Your code here
		 */
		for (String query : tfQuery.keySet()) {
			String queryL = query.toLowerCase();
			double idf;
			if (idfs.get(queryL) == null){
				idf = Math.log(idfs.get(LoadHandler.TotalCount) + 1);
			}else{
				idf = idfs.get(queryL);
			}
//			System.out.println("queryL: " + queryL + " idf " + idf);
			Double docTf = 0.0;
			docTf += urlweight * tfs.get("url").get(queryL);
			docTf += titleweight * tfs.get("title").get(queryL);
			docTf += bodyweight * tfs.get("body").get(queryL);
			docTf += headerweight * tfs.get("header").get(queryL);
			docTf += anchorweight * tfs.get("anchor").get(queryL);
			
			score += docTf * idf / (k1 + docTf);
		}
		
		score += pageRankLambda * pagerankScores.get(d);
		
//		System.out.println(d);
//		System.out.println("score = " + score);

		return score;
	}

	//do bm25 normalization
	public void normalizeTFs(Map<String,Map<String, Double>> tfs,Document d, Query q, Map<String,Double> tfQuery)
	{
		/*
		 * @//TODO : Your code here
		 */

		for (String queryWord : tfQuery.keySet())
		{
			String queryL = queryWord.toLowerCase();
			tfs.get("url"   ).put(queryL, normalizeFieldTF("url", tfs.get("url").get(queryL), d));
			tfs.get("title" ).put(queryL, normalizeFieldTF("title", tfs.get("title").get(queryL), d));
			tfs.get("body"  ).put(queryL, normalizeFieldTF("body", tfs.get("body").get(queryL), d));
			tfs.get("header").put(queryL, normalizeFieldTF("header", tfs.get("header").get(queryL), d));
			tfs.get("anchor").put(queryL, normalizeFieldTF("anchor", tfs.get("anchor").get(queryL), d));
		}
			
	}
	
	public double normalizeFieldTF(String field, double tf, Document doc){
		double normalizeTf = 0.0;
		if(tf > 0){
			switch (field) {
			case "url":
				normalizeTf = tf / (1 + burl * (lengths.get(doc).get("url") / avgLengths.get("url") - 1));
				break;
			case "title":
				normalizeTf = tf / (1 + btitle * (lengths.get(doc).get("title") / avgLengths.get("title") - 1));
				break;
			case "body":
				normalizeTf = tf / (1 + bbody * (lengths.get(doc).get("body") / avgLengths.get("body") - 1));
				break;
			case "header":
				normalizeTf = tf / (1 + bheader * (lengths.get(doc).get("header") / avgLengths.get("header") - 1));
				break;
			case "anchor":
				normalizeTf = tf / (1 + banchor * (lengths.get(doc).get("anchor") / avgLengths.get("anchor") - 1));
				break;
			}
		}
//		System.out.println("normalizeTF for field " + field + " tf " + tf + " is " + normalizeTf);
		return normalizeTf;
	}

	
	@Override
	public double getSimScore(Document d, Query q) 
	{
		
		Map<String,Map<String, Double>> tfs = this.getDocTermFreqs(d,q);
		
		Map<String,Double> tfQuery = getQueryFreqs(q);
		
		this.normalizeTFs(tfs, d, q, tfQuery);
		
        return getNetScore(tfs,q,tfQuery,d);
	}

	
	
	
}
