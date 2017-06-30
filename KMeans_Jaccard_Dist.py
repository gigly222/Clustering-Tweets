from nltk.corpus import stopwords
import re, string
MAXITERATIONS = 25  # Will make the program cut off eventually so that it doesn't run forever


# Calculates the Jaccard distance of a tweet using set notation
def jaccard_distance(setOne, setTwo):
    return 1 - float(len(setOne.intersection(setTwo))) / float(len(setOne.union(setTwo)))


def create_bag_of_words(tweet_text):
    stop_words = stopwords.words("english")
    tweets_lower_case = tweet_text.lower()
    line = tweets_lower_case.split(" ")
    sentence = []
    regex = re.compile("[%s]" % re.escape(string.punctuation))
    for word in line:
        word = word.strip()
        if not re.match(r'^https?:\/\/.*[\r\n]*', word) and word != '' and not re.match('\s', word) and word != 'rt' and not re.match('^@.*', word) and word not in stop_words:
            clean_word = regex.sub("", word)
            sentence.append(clean_word)
    return sentence


def update_clusters(tweets, clusters, id_with_clusters, jaccard_matrix, num_centroids):
    k = num_centroids

    # Initialize new clusters
    updated_clusters, updated_id_with_clusters = {}, {}  # This is an empty dictionary. Can look up values

    for k in range(k):
        updated_clusters[k] = set() # Create a new set to hold updated clusters

    for tweet1 in tweets:  # For all tweets

        min_distance = float("inf") # inifite distance. Want to minimize this distance metric using jaccard distance
        min_cluster = id_with_clusters[tweet1] #Get cluster the Tweet currently belongs too

        # Calculate min average distance to each cluster
        for k in clusters:  # for each cluster (25 is default)
            distance, total = 0, 0
            for tweet2 in clusters[k]: # For tweets associated with that cluster
                distance += jaccard_matrix[tweet1][tweet2]  # get jaccard distance pair for that tweet pair
                total += 1
            if total > 0:
                average_distance = float(distance / float(total))  # Calculated the average distance between all tweets in that cluster
                if min_distance > average_distance:
                    min_distance = average_distance
                    min_cluster = k # get new min cluster
        updated_id_with_clusters[tweet1] = min_cluster
        updated_clusters[min_cluster].add(tweet1)  # Per tweet, should have in cluster with jaccard distance minimized

    return updated_clusters, updated_id_with_clusters


def find_stable_clusters(tweets, clusters, id_with_clusters, jaccard_table, max_iterations, num_centroids):

    # Initialize previous cluster to compare changes with new clustering
    updated_clusters, updated_id_with_clusters = update_clusters(tweets, clusters, id_with_clusters, jaccard_table, num_centroids)

    clusters = updated_clusters
    id_with_clusters = updated_id_with_clusters

    # Converges until old and new iterations are the same
    iterations = 1
    print("MAX iterations ", max_iterations)
    while iterations < max_iterations:
        updated_clusters, updated_id_with_clusters = update_clusters(tweets, clusters, id_with_clusters, jaccard_table, num_centroids)
        iterations += 1
        if id_with_clusters != updated_id_with_clusters: #While these lists are not the same, we keep going.
            clusters = updated_clusters
            id_with_clusters = updated_id_with_clusters
        else:
            print("Converged at ", iterations , " iterations")
            return clusters, id_with_clusters

    # If meets max iteration, cut off
    print("Meet Max iterations")
    return clusters, id_with_clusters


# Prints cluster ID and tweet IDs for that cluster
def print_clusters(clusters):
    for k in clusters:
        line = str(k) + '\t'
        for tweetID in clusters[k]:
            line += str(tweetID) + ", "
        print(line)


def initialize_clusters(tweets, seeds, numCentroids):
    clusters, id_with_clusters = {}, {}  # Stores Cluster Points (tweet IDs...) and ID are stored in array with associated cluster. EX: idWithCluster[325409910827401217] = 23

    # Initially all tweets are assigned to no clusters
    for ID in tweets: id_with_clusters[ID] = -1000 #No cluster is -1000

    # Initialize clusters with seeds from file
    k = numCentroids
    for k in range(k): #for range 0 to number of clusters
        clusters[k] = set([seeds[k]]) # set each cluster to be one of the seeds ID #Cluster to tweet id. (Not random like part 1 K means)
        id_with_clusters[seeds[k]] = k # seeds[k] is a ID number. Look up that ID number and assign it to that cluster
    return clusters, id_with_clusters


# Calculate the Jaccard Distance between each tweet pair. Speeds up calculations by only doing it once.
def initialize_jaccard_table(tweets):
    jaccard_table = {}
    # For each tweet in tweet list
    for tweetOne in tweets:
        jaccard_table[tweetOne] = {}  # Create empty list for this tweetOne's distances and add to Jaccard Table
        tweet_bag_words_one = set(create_bag_of_words(tweets[tweetOne]["text"]))  # Returns bag of words for the tweet. (Pass in only text, not ID)
        # Compare to all other tweets
        for tweetTwo in tweets:
            if tweetTwo not in jaccard_table:
                jaccard_table[tweetTwo] = {}  # Create another empty list for this new Tweet and add to Jaccard Table
            tweet_bag_words_two = set(create_bag_of_words(tweets[tweetTwo]["text"]))  # Returns bag of words for the new tweet
            jaccard_dist = jaccard_distance(tweet_bag_words_one, tweet_bag_words_two)  # Gets jaccard distance between tweets (0 means the same words, 1 means totally different/no shared words)
            jaccard_table[tweetOne][tweetTwo], jaccard_table[tweetTwo][tweetOne] = jaccard_dist, jaccard_dist

    return jaccard_table


# Initilize lookup table of distance and cluster dictionaries
def kMeans_set_up(tweets, seeds, num_centroids):
    jaccard_table = initialize_jaccard_table(tweets)
    clusters, id_with_clusters = initialize_clusters(tweets, seeds, num_centroids)

    return jaccard_table, clusters, id_with_clusters


# Run Kmeans Algorithm on Tweets using Jaccard Distance
def kMeans(seeds, tweets, num_centroids):

    # Initilize a Lookup Table to calculate all the Jaccard distance pairs. Initilize all clusters
    jaccard_table, clusters, id_with_clusters = kMeans_set_up(tweets, seeds, num_centroids)

    # Run Kmeans algorithm
    clusters, id_with_clusters = find_stable_clusters(tweets, clusters, id_with_clusters, jaccard_table, MAXITERATIONS, num_centroids)

    # Print results to screen
    print_clusters(clusters)

    print("Checking to see if similar text is in the same cluster...")
    print("Cluster ID : 323909308188344320 and Cluster ID : 324229792834674689 in same cluster. " )
    print("ID : 323909308188344320 Text: ", tweets[323909308188344320]['text'])
    print("ID : 324229792834674689 Text: ", tweets[324229792834674689]['text'])

    return clusters, id_with_clusters