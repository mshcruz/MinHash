import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.Set;

public class MinHash<T> {

    private int hash[][];
    private int numHash;
    
    /**
     * 
     */
    public MinHash(int numHash){
	this.numHash = numHash;

	System.out.println("numHash: "+numHash);
    }
    
    private void printMinHashValues(int[][] minHashValues){
	for(int i=0; i < 2; i++){
	    for(int j=0; j<numHash; j++){
		System.out.print(minHashValues[i][j]+" ");
	    }
	    System.out.println();
	}
    }

    public double similarity(Set<T> set1, Set<T> set2){

        int numSets = 2;
        Map<T, boolean[]> bitMap = buildBitMap(set1, set2);
        hash = new int[bitMap.size()][numHash];        
	
	Random r = new Random(11);


	for (int j = 0; j < bitMap.size(); j++){
	    for (int i = 0; i < numHash; i++){
		int a = (int)r.nextInt();
		int b = (int)r.nextInt();
		int c = (int)r.nextInt();
		int x = hash(a*b*c, a, b, c);
		hash[j][i] = x;
		//		System.out.print(hash[j][i]+" ");
	    }  
	    //	    System.out.println();
	    //	    System.out.println();
        } 

        int[][] minHashValues = initializeHashBuckets(numSets, numHash);
	//	printMinHashValues(minHashValues);
	//	System.out.println();
        computeMinHashForSet(set1, 0, minHashValues, bitMap);
	//	printMinHashValues(minHashValues);
	//	System.out.println();
        computeMinHashForSet(set2, 1, minHashValues, bitMap);
	//	printMinHashValues(minHashValues);
	//	System.out.println();
        return computeSimilarityFromSignatures(minHashValues, numHash);
    }
    
    /**
     * 
vv     */
    private static int[][] initializeHashBuckets(int numSets, int numHashFunctions) {
	int[][] minHashValues = new int[numSets][numHashFunctions];

        for (int i = 0; i < numSets; i++) {
	    for (int j = 0; j < numHashFunctions; j++) {
		minHashValues[i][j] = Integer.MAX_VALUE;
            }
        }
        return minHashValues;
    }
     
    /**
     * 
     * @param minHashValues
     * @param numHashFunctions
     * @return
     */
    private static double computeSimilarityFromSignatures(int[][] minHashValues, int numHashFunctions) {
	int identicalMinHashes = 0;
        for (int i = 0; i < numHashFunctions; i++){
            if (minHashValues[0][i] == minHashValues[1][i]) {
                identicalMinHashes++;
            }
        }
	System.out.println("IdenticalMinhashes: "+identicalMinHashes);
        return (1.0 * identicalMinHashes) / numHashFunctions;
    }
    
    /**
     * 
     * @param x
     * @param a
     * @param b
     * @param c
     * @return
     */
    private static int hash(int x, int a, int b, int c) {
        int hashValue = (int)((a * (x >> 4) + b * x + c) & 131071);
        return Math.abs(hashValue);
    }

    private void computeMinHashForSet(Set<T> set, int setIndex, int[][] minHashValues, Map<T, boolean[]> bitArray){

	int index=0;
	
	for(T element : bitArray.keySet()) { // for every element in the bit array
	    if(set.contains(element)) { // if the set contains the element
		for (int i = 0; i < numHash; i++){ // for every hash
		    int hindex = hash[index][i]; // get the hash
		    if (hindex < minHashValues[setIndex][i]) { 
			// if current hash is smaller than the existing hash in the slot then replace with the smaller hash value
			minHashValues[setIndex][i] = hindex;
		    }
		}
	    }
	    index++;
	}
    }
    
    /**
     * 
     * @param set1
     * @param set2
     * @return
     */
    public Map<T,boolean[]> buildBitMap(Set<T> set1, Set<T> set2){
	
	Map<T,boolean[]> bitArray = new HashMap<T,boolean[]>();
	
	for(T t : set1){
	    bitArray.put(t, new boolean[]{true,false});
	}
	
	for(T t : set2){
	    if(bitArray.containsKey(t)){
		// item is present in set1
		bitArray.put(t, new boolean[]{true,true});
	    }else if(!bitArray.containsKey(t)){
		// item is not present in set1
		bitArray.put(t, new boolean[]{false,true});
	    }
	}
	return bitArray;
    }
    
    
    public static void main(String[] args){
	Set<String> set1 = new HashSet<String>();
	set1.add("b");
	set1.add("f");
	set1.add("g");
	set1.add("k");
	
	Set<String> set2 = new HashSet<String>();
	set2.add("a");
	set2.add("b");
	set2.add("c");
	set2.add("g");
	set2.add("j");

	Set<String> set3 = new HashSet<String>();
	set3.add("d");
	set3.add("g");
	set3.add("h");
	set3.add("j");

	Set<String> set4 = new HashSet<String>();
	set4.add("b");
	set4.add("c");
	set4.add("e");
	set4.add("f");
	set4.add("g");
	set4.add("k");

	MinHash<String> minHash = new MinHash<String>(1000);
	System.out.println("Similarity S1 and S2: "+minHash.similarity(set1, set2)+" Real similarity: 0.28");

	MinHash<String> minHash2 = new MinHash<String>(1000);
	System.out.println("Similarity S1 and S3: "+minHash2.similarity(set1, set3)+" Real similarity: 0.14");

	MinHash<String> minHash3 = new MinHash<String>(1000);
	System.out.println("Similarity S1 and S4: "+minHash3.similarity(set1, set4)+" Real similarity: 0.66");

	MinHash<String> minHash4 = new MinHash<String>(1000);
	System.out.println("Similarity S2 and S3: "+minHash4.similarity(set2, set3)+" Real similarity: 0.28");

	MinHash<String> minHash5 = new MinHash<String>(1000);
	System.out.println("Similarity S2 and S4: "+minHash5.similarity(set2, set4)+" Real similarity: 0.42");
	
	MinHash<String> minHash6 = new MinHash<String>(1000);
	System.out.println("Similarity S3 and S4: "+minHash6.similarity(set3, set4)+" Real similarity: 0.11");
    }
}
