package cs276.pa4;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

//doesn't necessarily have to use task 2 (could use task 1, in which case, you'd probably like to extend CosineSimilarityScorer instead)
public class SmallestWindowScorer extends BM25Scorer
{

	/////smallest window specifichyperparameters////////
    double B = 2;    	    
    double boostmod = -1;
    Map<Document, Map<Query, Double>> docWindows = new HashMap<Document, Map<Query, Double>>();
    
    //////////////////////////////
	
	public SmallestWindowScorer(Map<String, Double> idfs,Map<Query, List<Document>> queryDict) 
	{
		super(idfs, queryDict);
		handleSmallestWindow();
	}

	
	public void handleSmallestWindow()
	{
		/*
		 * @//TODO : Your code here
		 */		
		for (Query query : queryDict.keySet()){
			for (Document doc : queryDict.get(query)){
    			double curSmallestWindow = Double.POSITIVE_INFINITY;
    			
    			// url
    			curSmallestWindow = checkWindow(query, doc.url, curSmallestWindow, false);
    			
    			// title
    			curSmallestWindow = checkWindow(query, doc.title, curSmallestWindow, false);

    			// body
    			if (doc.body_hits != null) {
    				curSmallestWindow = checkWindowBody(query, doc.body_hits, curSmallestWindow);
    			}
    			
    			// header
    			if (doc.headers != null){
    				for (String header : doc.headers){
    	    			curSmallestWindow = checkWindow(query, header, curSmallestWindow, false);
    				}
    			}
 
    			// anchors
    			if (doc.anchors != null){
    				for (String anchorText : doc.anchors.keySet()){
    	    			curSmallestWindow = checkWindow(query, anchorText, curSmallestWindow, false);
    				}
    			}
    			
    			if (!docWindows.containsKey(doc)){
    				docWindows.put(doc, new HashMap<Query, Double>());
    			}
    			
    			docWindows.get(doc).put(query, curSmallestWindow);
    		}
		}
					
		
	}
	
	private double checkWindowBody(Query q, Map<String, List<Integer>> bodyHits, double curSmallestWindow) {
		double smallestWindow = Double.POSITIVE_INFINITY;
		List<String> queryWords = q.words;
		int queryLength = queryWords.size();
		
//		System.out.println("query = " + queryWords);
//		System.out.println("posting = " + bodyHits);
		
		// convert query words to lower case
		for (int i = 0; i < queryLength; i++) {
			String word = queryWords.get(i).toLowerCase();;
			queryWords.set(i, word);
		}
		
		// convert doc terms to lower case
		for (String term : bodyHits.keySet()) {
			bodyHits.put(term.toLowerCase(), bodyHits.get(term));
		}
		
		// if not all terms appear in the doc
		for (String term : queryWords) {
			if (!bodyHits.containsKey(term)) {
				return curSmallestWindow;
			}
		}
		
		// traverse through posting list, calculate smallest window size
		// record the current positions
		Map<String, Integer> curPositions = new HashMap<String, Integer>();
		for (String term : queryWords) {
			curPositions.put(term, 0);
		}
		while (true) {
			String smallestPosTerm = getSmallestPositionTerm(bodyHits, curPositions);
			int smallestPos = curPositions.get(smallestPosTerm);
			
			String largestPosTerm = getLargestPositionTerm(bodyHits, curPositions);
			int largestPos = curPositions.get(largestPosTerm);
			
			double begin = bodyHits.get(smallestPosTerm).get(smallestPos);
			double end = bodyHits.get(largestPosTerm).get(largestPos);
			double window = end - begin + 1;
			if (window < smallestWindow) smallestWindow = window;
			
			// already hit the end
			if (smallestPos >= bodyHits.get(smallestPosTerm).size() - 1) {
				break;
			}
			
			// advance the pointer
			curPositions.put(smallestPosTerm, smallestPos+1);
		}

//		System.out.println("curWindow = " + Double.toString(curSmallestWindow));
//		System.out.println("window = " + Double.toString(smallestWindow));
		return (smallestWindow < curSmallestWindow) ? 4*smallestWindow : curSmallestWindow;
	}
	
	//  get the largest position term in the current pointers
	private String getLargestPositionTerm(Map<String, List<Integer>> bodyHits, Map<String, Integer> curPositions) {
		int largest = -1;
		String largestTerm = "";
		for (String term : curPositions.keySet()) {
			int idx = curPositions.get(term);
			int pos = bodyHits.get(term).get(idx);
			if (pos > largest) {
				largest = pos;
				largestTerm = term;
			}
		}
		return largestTerm;
	}
	
	//  get the smallest position term in the current pointers
	private String getSmallestPositionTerm(Map<String, List<Integer>> bodyHits, Map<String, Integer> curPositions) {
		int smallest = Integer.MAX_VALUE;
		String smallestTerm = "";
		for (String term : curPositions.keySet()) {
			int idx = curPositions.get(term);
			int pos = bodyHits.get(term).get(idx);
			if (pos < smallest) {
				smallest = pos;
				smallestTerm = term;
			}
		}
		return smallestTerm;
	}

	
	public double checkWindow(Query q,String docstr,double curSmallestWindow,boolean isBodyField)
	{
		/*
		 * @//TODO : Your code here
		 */
		// query word to lower case?
//		System.out.println("query: " + q.queryWords + " docstr: " + docstr);
//		System.out.println("at the beginning, curSmallestWindow == " + curSmallestWindow);
		Set<String> querywordSet = new HashSet<String>();
		querywordSet.addAll(q.words);
		double smallestWindow = Double.POSITIVE_INFINITY;
		
		if (!isBodyField){
			String[] tokens = docstr.split("\\W+");
			List<Integer> querywordLocation = new ArrayList<Integer>();
//			Set<String> querywordInDoc = new HashSet<String>();
			Map<String, Integer> querywordCountInDoc = new HashMap<String, Integer>();
			for (String queryword : querywordSet){
				querywordCountInDoc.put(queryword, new Integer(0));
			}
			
			int beginIndex = Integer.MIN_VALUE;
			int endIndex = Integer.MAX_VALUE;
			int i = 0;
			while (i < tokens.length && !querywordSet.contains(tokens[i])){
				i++;
			}
			
			// check the whole docstr string, unable to find a query word, just return
			if (i >= tokens.length){ 
				return curSmallestWindow;
			}
			
			beginIndex = i;
			querywordLocation.add(new Integer(i));
			querywordCountInDoc.put(tokens[i], querywordCountInDoc.get(tokens[i]) + 1);
			
			for (i = beginIndex + 1; i < tokens.length; i++){
				if (querywordSet.contains(tokens[i])){
//					querywordInDoc.add(tokens[i]);
					querywordLocation.add(new Integer(i));
					querywordCountInDoc.put(tokens[i], querywordCountInDoc.get(tokens[i]) + 1);
					if (checkWindowStatus(querywordCountInDoc) > 0){
						// found the first minimum window
						endIndex = i;
						
						// move the beginIndex to right until there is no duplicates of the query word in the current window
						while(querywordCountInDoc.get(tokens[beginIndex]).intValue() > 1){
							querywordCountInDoc.put(tokens[beginIndex], querywordCountInDoc.get(tokens[beginIndex]) - 1);
							beginIndex = querywordLocation.remove(0).intValue(); // next location of query word in docstr
						}
						
						int minimumWindow = endIndex - beginIndex + 1;
						if (minimumWindow < smallestWindow){
							smallestWindow = minimumWindow;
						}
						break;
					}
				}
			}
			
			// check the whole docstr string, unable to find a window containing all the query words, just return
			if (i >= tokens.length){
				return curSmallestWindow;
			}
			
			for (i = endIndex + 1; i < tokens.length; i++){
				if (querywordSet.contains(tokens[i])){
//					querywordInDoc.add(tokens[i]);
					querywordLocation.add(new Integer(i));
					querywordCountInDoc.put(tokens[i], querywordCountInDoc.get(tokens[i]) + 1);
					endIndex = i; // update endIndex pointer
					
					// move the beginIndex to right until there is no duplicates of the query word in the current window
					while(querywordCountInDoc.get(tokens[beginIndex]).intValue() > 1){
						querywordCountInDoc.put(tokens[beginIndex], querywordCountInDoc.get(tokens[beginIndex]) - 1);
						beginIndex = querywordLocation.remove(0).intValue(); // next location of query word in docstr
					}
					
					int minimumWindow = endIndex - beginIndex + 1;
					if (minimumWindow < smallestWindow){
						smallestWindow = minimumWindow;
					}	
				}
			}
			
			// update window
			if (smallestWindow < curSmallestWindow){
//				System.out.println("update smallest window to " + smallestWindow);
				return smallestWindow;
			}
			
			
		}
		return curSmallestWindow;
	}
	
	private int checkWindowStatus(Map<String, Integer> querywordCountInDoc){
		int maxCount = 0;
		for (String queryword : querywordCountInDoc.keySet()){
			if (querywordCountInDoc.get(queryword).intValue() == 0){
//				System.out.println(queryword);
				return 0; // there is still at least a query word not in the current window
			}else if (querywordCountInDoc.get(queryword).intValue()  > maxCount){
				maxCount = querywordCountInDoc.get(queryword);
			}
		}
//		System.out.println(maxCount);
		return maxCount; // the maximum count for a query word in the current window (with all the query words in the current window) 
	}
	
	
	
	@Override
	public double getSimScore(Document d, Query q) {
		Map<String,Map<String, Double>> tfs = this.getDocTermFreqs(d,q);
			
		Map<String,Double> tfQuery = getQueryFreqs(q);
		
		this.normalizeTFs(tfs, d, q, tfQuery);
		
		double netScore = getNetScore(tfs,q,tfQuery,d);
//		System.out.println("previous score: " + netScore);

		// aggregate smallest window
		netScore = netScore * smoothWindowFactor(docWindows.get(d).get(q), tfQuery.size());
//		System.out.println("querySize: " + tfQuery.size() + " smoothWindowFactor " + smoothWindowFactor(docWindows.get(d).get(q), tfQuery.size()));
//		System.out.println("now score: " + netScore + " windows: " + docWindows.get(d).get(q));
		return netScore;
	}

	public double smoothWindowFactor(double window, int queryLength){
		if(boostmod < 0){
			double test = (B-1) * Math.exp(boostmod * (window - queryLength)) + 1;
//			System.out.println("test1: queryLength " + queryLength + "window " + window + " smoothscore " + test);
			return (double) (B-1) * Math.exp(boostmod * (window - queryLength)) + 1; // decrease exponentially, B should be smaller than e
		}else{
			double test = queryLength * (B-1) / window + 1;
//			System.out.println("test2: queryLength " + queryLength + "window " + window + " smoothscore " + test);
			return (double) queryLength * (B-1) / window + 1; // decrease as 1/x
		}
	}
}
