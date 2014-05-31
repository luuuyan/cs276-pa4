package cs276.pa4;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;




public class AScorer 
{
	
	Map<String,Double> idfs;
	static String[] TFTYPES = {"url","title","body","header","anchor"};
	int smoothingBodyLength = 500;
	
	public AScorer(Map<String,Double> idfs)
	{
		this.idfs = idfs;
	}
	
	//scores each document for each query
//	public abstract double getSimScore(Document d, Query q);
	
	//handle the query vector
	public Map<String,Double> getQueryFreqs(Query q)
	{
		Map<String,Double> tfQuery = new HashMap<String,Double>();
		
		/*
		 * @//TODO : 
		 */
		for (String term : q.words){
			String termL = term.toLowerCase();
			if (termL.length() > 0){
				if (tfQuery.containsKey(termL)){
					tfQuery.put(termL, tfQuery.get(termL) + 1);
				}else {
					tfQuery.put(termL, 1.0);
				}
			}
		}
		
//		// calculate log term frequency
//		for (String term : tfQuery.keySet()) {
//			tfQuery.put(term, 1 + Math.log(tfQuery.get(term)));
//		}
		
		return tfQuery;
	}
	

	
	////////////////////Initialization/Parsing Methods/////////////////////

	// parse headers
	private static Map<String, Integer> getHeaderTermFreqs(List<String> headers) {
		Map<String, Integer> headerTfs = new HashMap<String, Integer>();
		for (String header : headers) {
			String[] headerTokens = header.split("\\W+");
			for (String token : headerTokens){
				String tokenL = token.toLowerCase();
				if (token.length() > 0){
					if (headerTfs.containsKey(tokenL)){
						headerTfs.put(tokenL, headerTfs.get(tokenL) + 1);
					}else { 
						headerTfs.put(tokenL, 1);
					}
				}
			}
		}
		return headerTfs;
	}
	
	// parse url or title
	private static Map<String, Integer> getGenTermFreqs(String s){
		Map<String, Integer> Tfs = new HashMap<String, Integer>();
		String[] tokens = s.split("\\W+");
		for (String token : tokens){
			String tokenL = token.toLowerCase();
			if (token.length() > 0){
				if (Tfs.containsKey(tokenL)){
					Tfs.put(tokenL, Tfs.get(tokenL) + 1);
				}else { 
					Tfs.put(tokenL, 1);
				}
			}
		}
		return Tfs;
	}

	// parse anchor
	private static Map<String, Integer> getAnchorTermFreqs(Map<String, Integer> anchors){
		Map<String, Integer> Tfs = new HashMap<String, Integer>();
		for (String anchor : anchors.keySet()){
			int anchorCount = anchors.get(anchor);
			String[] tokens = anchor.split("\\W+");
			for (String token : tokens){
				String tokenL = token.toLowerCase();
				if (token.length() > 0){
					if (Tfs.containsKey(tokenL)){
						Tfs.put(tokenL, Tfs.get(tokenL) + anchorCount);
					}else { 
						Tfs.put(tokenL, anchorCount);
					}
				}
			}
		}
		return Tfs;
	}
	
	// parse body hits
	private static Map<String, Integer> getBodyTermFreqs(Map<String, List<Integer>> body_hits){
		Map<String, Integer> Tfs = new HashMap<String, Integer>();
		for (String term : body_hits.keySet()){
			int count = body_hits.get(term).size();
			Tfs.put(term.toLowerCase(), count);
		}
		return Tfs;
	}
	
    ////////////////////////////////////////////////////////
	
	
	/*/
	 * Creates the various kinds of term frequences (url, title, body, header, and anchor)
	 * You can override this if you'd like, but it's likely that your concrete classes will share this implementation
	 */
	public static Map<String,Map<String, Double>> getDocTermFreqs(Document d, Query q)
	{
		//map from tf type -> queryWord -> score
		Map<String,Map<String, Double>> tfs = new HashMap<String,Map<String, Double>>();
		
		////////////////////Initialization/////////////////////

		Map<String, Integer> urlTfs = new HashMap<String, Integer>();
		Map<String, Integer> titleTfs = new HashMap<String, Integer>();
		Map<String, Integer> headerTfs = new HashMap<String, Integer>();
		Map<String, Integer> anchorTfs = new HashMap<String, Integer>();
		Map<String, Integer> bodyTfs = new HashMap<String, Integer>();
		
		if (d.url != null)
			urlTfs = getGenTermFreqs(d.url);
		if (d.title != null)
			titleTfs = getGenTermFreqs(d.title);
		if (d.headers != null)	
			headerTfs = getHeaderTermFreqs(d.headers);
		if (d.anchors != null)
			anchorTfs = getAnchorTermFreqs(d.anchors);
		if (d.body_hits != null)
			bodyTfs = getBodyTermFreqs(d.body_hits);
		
		for (int i = 0; i < TFTYPES.length; i++) {
			tfs.put(TFTYPES[i], new HashMap<String, Double>());
		}
		
		//////////handle counts//////
				
		//loop through query terms increasing relevant tfs
		for (String queryWord : q.words)
		{
			String queryL = queryWord.toLowerCase();
			tfs.get("url"   ).put(queryL, getTfFromDoc(urlTfs, queryL));
			tfs.get("title" ).put(queryL, getTfFromDoc(titleTfs, queryL));
			tfs.get("body"  ).put(queryL, getTfFromDoc(bodyTfs, queryL));
			tfs.get("header").put(queryL, getTfFromDoc(headerTfs, queryL));
			tfs.get("anchor").put(queryL, getTfFromDoc(anchorTfs, queryL));
		}
		return tfs;
	}
	
	private static double getTfFromDoc(Map<String, Integer> map, String term) {
		if (map.containsKey(term)) {
			return (double)map.get(term);
		} else {
			return 0.0;
		}
	}
	
	// Note: add two new functions for pa4
	public static double[] getTfIdf(Document d, Query q,  Map<String, Double> idfs)
	{
		// tdIdf stores tdIdf w.r.t to 5 fields using raw td
		double[] tdIdf = new double[5];

		if (d.url != null)
			tdIdf[0] = getTfIdfForField(getGenTermFreqs(d.url),idfs,q);
		if (d.title != null)
			tdIdf[1] = getTfIdfForField(getGenTermFreqs(d.title),idfs,q);
		if (d.headers != null)	
			tdIdf[2] = getTfIdfForField(getHeaderTermFreqs(d.headers),idfs,q);
		if (d.anchors != null)
			tdIdf[3] = getTfIdfForField(getAnchorTermFreqs(d.anchors),idfs,q);
		if (d.body_hits != null)
			tdIdf[4] = getTfIdfForField(getBodyTermFreqs(d.body_hits),idfs,q);

		return tdIdf;
	}
	
	public static double getTfIdfForField(Map<String, Integer> tfFromDocField, Map<String, Double> dfs, Query q){
		double score = 0.0;
		for (String queryword : q.words) {
			String queryL = queryword.toLowerCase();
			double idf;
			if (dfs.get(queryL) == null){
				idf = dfs.size() + 1; // small modification, need to check
			}else{
				idf = (double) (dfs.size() + 1) / (dfs.get(queryL) + 1);
			}
			
			if (tfFromDocField.containsKey(queryL)){
//				score += (double) Math.log(1 + tfFromDocField.get(queryL)) * Math.log(idf);
				score += (double) Math.log(1 + tfFromDocField.get(queryL)) * idf;

			}
		}
		return score;
	}

	
	private double tfSmooth(double tf, Document d) {
		int length = d.body_length + smoothingBodyLength;

		if (tf > 0)
			return (1 + Math.log(tf)) / length;
		else
			return 0;
	}
}
