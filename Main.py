import sys
import json
from KMeans_Jaccard_Dist import kMeans


# Writes cluster ID and tweet IDs for that cluster to file
def write_clusters_to_file(output_file, clusters):
    for k in clusters:
        write_line = str(k) + '\t'
        for tweetID in clusters[k]:
            write_line += str(tweetID) + ", "
        output_file.write(write_line + "\n")

''' MAIN FUNCTION '''
if __name__ == "__main__":

    # check to see if argument input is the right length
    if len(sys.argv) != 5:
        print("Not enough input provided!")
        sys.exit(0)

    # Read in Files
    num_centroids, seed_path, tweets_path, output_path = sys.argv[1], sys.argv[2], sys.argv[3],  sys.argv[4]  # k-means <numberOfClusters> <input-file-name> <output-file-name>
    print("Number of Centroids : ", int(num_centroids))
    tweets = {}
    with open(tweets_path, 'r') as f:
        for line in f:
            tweet = json.loads(line)
            tweets[tweet['id']] = tweet
    seeds =[]
    with open(seed_path) as sf:
        for line in sf:
            seeds.append(int(line.rstrip(',\n')))
    sf.close()

    # Run KMeans
    clusters, id_with_clusters = kMeans(seeds, tweets, int(num_centroids))

    # Write to file
    output_file = open(output_path, "w")
    write_clusters_to_file(output_file, clusters)

    output_file.close()