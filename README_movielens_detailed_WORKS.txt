
recommender POC 

https://www.data-mania.com/blog/how-to-build-a-recommendation-engine-in-r/

Demo: How to build a recommendation engine in R

Here is the link to the dataset used in the demo: http://files.grouplens.org/datasets/movielens/ml-latest-small.zip

So, how to build a recommendation engine in R… starting with the reading step in R, let’s read-in all our datasets and build a ratings matrix

#NOTE:requires external files


#STEP1) install packages

install.packages("recommenderlab")
install.packages("reshape2")
install.packages("ggplot2")
install.packages("stringi")
install.packages("dplyr")

library(recommenderlab)
library(reshape2)
library(ggplot2)
library(stringi)
library(dplyr)

getwd()
setwd("/Users/hivan/work/recommender_POC")

# Read training file along with header
movies=read.csv("movies.csv")
links=read.csv("links.csv")
ratings=read.csv("ratings.csv")
tags=read.csv("tags.csv")

> head(movies)
  movieId                   title             genres
1       1                   Toy Story (1995)  Adventure|Animation|Children|Comedy|Fantasy
2       2                     Jumanji (1995)  Adventure|Children|Fantasy
3       3            Grumpier Old Men (1995)  Comedy|Romance
4       4           Waiting to Exhale (1995)  Comedy|Drama|Romance
5       5 Father of the Bride Part II (1995)  Comedy
6       6                        Heat (1995)  Action|Crime|Thriller


> head(links)
  movieId imdbId tmdbId
1       1 114709    862
2       2 113497   8844
3       3 113228  15602
4       4 114885  31357
5       5 113041  11862
6       6 113277    949


> head(ratings)
  userId movieId rating timestamp
1      1       1      4 964982703
2      1       3      4 964981247
3      1       6      4 964982224
4      1      47      5 964983815
5      1      50      5 964982931
6      1      70      3 964982400


> head(tags)
  userId movieId             tag  timestamp
1      2   60756           funny 1445714994
2      2   60756 Highly quotable 1445714996
3      2   60756    will ferrell 1445714992
4      2   89774    Boxing story 1445715207
5      2   89774             MMA 1445715200
6      2   89774       Tom Hardy 1445715205


#STEP2) Create dataset

#Create ratings matrix with rows as users and columns as movies. We don't need timestamp
ratingmat = dcast(ratings, userId~movieId, value.var = "rating", na.rm=FALSE)


#We can now remove user ids
ratingmat = as.matrix(ratingmat[,-1])

#Convert ratings matrix to real rating matrx which makes it dense<
ratingmat = as(ratingmat, "realRatingMatrix")

#Normalize the ratings matrix
ratingmat = normalize(ratingmat)



#STEP3) Build recommender

#Create Recommender Model. The parameters are UBCF and Cosine similarity. We take 10 nearest neighbours
rec_mod = Recommender(ratingmat, method = "UBCF", param=list(method="Cosine",nn=10))


#Obtain top 5 recommendations for 1st user entry in dataset
Top_5_pred = predict(rec_mod, ratingmat[1], n=5)

#Convert the recommendations to a list
Top_5_List = as(Top_5_pred, "list")
Top_5_List
[1] "58"   "867"  "1688" "4171" "5135"


#We convert the list to a dataframe and change the column name to movieId
Top_5_df=data.frame(Top_5_List)
colnames(Top_5_df)="movieId"

#Since movieId is of type integer in Movies data, we typecast id in our recommendations as well
Top_5_df$movieId=as.numeric(levels(Top_5_df$movieId))

#Merge the movie ids with names to get titles and genres
names=left_join(Top_5_df, movies, by="movieId")

> names
  movieId                                title    genres
1    1688                     Anastasia (1997)    Adventure|Animation|Children|Drama|Musical 
2    4171 Long Night's Journey Into Day (2000)    Documentary
3    5135               Monsoon Wedding (2001)    Comedy|Romance
4      58    Postman, The (Postino, Il) (1994)    Comedy|Drama|Romance
5     867                       Carpool (1996)    Comedy|Crime


save.image(file='r_image.RData')


########################## STOP HERE #############################




===> NOTE, this is corrupted, so need to do manually
> Top_5_df
  movieId
1    1688
2    4171
3    5135
4      58
5     867

grep 1688 movies.txt
1688,Anastasia (1997),Adventure|Animation|Children|Drama|Musical

grep 4171 movies.csv
4171,Long Night's Journey Into Day (2000),Documentary

grep 5135 movies.csv
5135,Monsoon Wedding (2001),Comedy|Romance

grep 58 movies.csv
58,"Postman, The (Postino, Il) (1994)",Comedy|Drama|Romance

grep 867 movies.csv
867,Carpool (1996),Comedy|Crime


==> NOTE: the last three have a similar comedy genre




save.image(file='r_image.RData')


############################################ STOP HERE #####################


# Remove 'id' column. We do not need it
tr<-tr[,-c(1)]


# Check, if removed
tr[tr$user==1,]

       user movie rating
34179     1  1907      4
64257     1  1287      5
68565     1  1566      4
71239     1   260      4
125237    1   919      4


# Using acast to convert above data as follows:
#       m1  m2   m3   m4
# u1    3   4    2    5
# u2    1   6    5
# u3    4   4    2    5

g<-acast(tr, user ~ movie)

# Check the class of g
class(g)


# Convert it as a matrix
R<-as.matrix(g)
 
# Convert R into realRatingMatrix data structure
#   realRatingMatrix is a recommenderlab sparse-matrix like data-structure
r <- as(R, "realRatingMatrix")

#view r
r
6040 x 3676 rating matrix of class ‘realRatingMatrix’ with 750156 ratings.

# view r in other possible ways
as(r, "list")     # A list
as(r, "matrix")   # A sparse matrix
 
# I can turn it into data-frame
head(as(r, "data.frame"))
 
# normalize the rating matrix
r_m <- normalize(r)

r_m
6040 x 3676 rating matrix of class ‘realRatingMatrix’ with 750156 ratings.
Normalized using center on rows.

as(r_m, "list")

# Draw an image plot of raw-ratings & normalized ratings
#  A column represents one specific movie and ratings by users
#   are shaded.
#   Note that some items are always rated 'black' by most users
#    while some items are not rated by many users
#     On the other hand a few users always give high ratings
#      as in some cases a series of black dots cut across items

image(r, main = "Raw Ratings")       
image(r_m, main = "Normalized Ratings")



#STEP2) Generate recommender

# Can also turn the matrix into a 0-1 binary matrix
r_b <- binarize(r, minRating=1)
as(r_b, "matrix")


# Create a recommender object (model)
#   Run anyone of the following four code lines.
#     Do not run all four
#       They pertain to four different algorithms.
#        UBCF: User-based collaborative filtering
#        IBCF: Item-based collaborative filtering
#      Parameter 'method' decides similarity measure
#        Cosine or Jaccard
rec=Recommender(r[1:nrow(r)],method="UBCF", param=list(normalize = "Z-score",method="Cosine",nn=5, minRating=1))
rec=Recommender(r[1:nrow(r)],method="UBCF", param=list(normalize = "Z-score",method="Jaccard",nn=5, minRating=1))
rec=Recommender(r[1:nrow(r)],method="IBCF", param=list(normalize = "Z-score",method="Jaccard",minRating=1))
rec=Recommender(r[1:nrow(r)],method="POPULAR")


# selected UBCF
> rec=Recommender(r[1:nrow(r)],method="UBCF", param=list(normalize = "Z-score",method="Cosine",nn=5, minRating=1))
Warning: Unknown parameters: minRating
Available parameter (with default values):
method	 =  cosine
nn	 =  25
sample	 =  FALSE
normalize	 =  center
verbose	 =  FALSE


>print(rec)
Recommender of type ‘UBCF’ for ‘realRatingMatrix’ 
learned using 6040 users.

> names(getModel(rec))
[1] "description" "data"        "method"      "nn"          "sample"     
[6] "normalize"   "verbose" 

> getModel(rec)$nn
[1] 5



#STEP3, generate predictions

############Create predictions#############################
# This prediction does not predict movie ratings for test.
#   But it fills up the user 'X' item matrix so that
#    for any userid and movieid, I can find predicted rating
#     dim(r) shows there are 6040 users (rows)
#      'type' parameter decides whether you want ratings or top-n items
#         get top-10 recommendations for a user, as:
#             predict(rec, r[1:nrow(r)], type="topNList", n=10)
recom <- predict(rec, r[1:nrow(r)], type="ratings")
recom


########## Examination of model & experimentation  #############
########## This section can be skipped #########################
 
# Convert prediction into list, user-wise
as(recom, "list")
# Study and Compare the following:
as(r, "matrix")     # Has lots of NAs. 'r' is the original matrix
as(recom, "matrix") # Is full of ratings. NAs disappear
as(recom, "matrix")[,1:10] # Show ratings for all users for items 1 to 10
as(recom, "matrix")[5,3]   # Rating for user 5 for item at index 3
as.integer(as(recom, "matrix")[5,3]) # Just get the integer value
as.integer(round(as(recom, "matrix")[6039,8])) # Just get the correct integer value
as.integer(round(as(recom, "matrix")[368,3717])) 
 

# Convert all your recommendations to list structure
rec_list<-as(recom,"list")
head(summary(rec_list))

# Access this list. User 2, item at index 2
rec_list[[2]][2]

# Convert to data frame all recommendations for user 1
u1<-as.data.frame(rec_list[[1]])
attributes(u1)
class(u1)

# Create a column by name of id in data frame u1 and populate it with row names
u1$id<-row.names(u1)
# Check movie ratings are in column 1 of u1
u1
# Now access movie ratings in column 1 for u1
u1[u1$id==3952,1]



# Read test file
test<-read.csv("test_v2.csv",header=TRUE)
head(test)
# Get ratings list
rec_list<-as(recom,"list")
head(summary(rec_list))
ratings<-NULL
# For all lines in test file, one by one
for ( u in 1:length(test[,2]))
{
   # Read userid and movieid from columns 2 and 3 of test data
   userid <- test[u,2]
   movieid<-test[u,3]
 
   # Get as list & then convert to data frame all recommendations for user: userid
   u1<-as.data.frame(rec_list[[userid]])
   # Create a (second column) column-id in the data-frame u1 and populate it with row-names
   # Remember (or check) that rownames of u1 contain are by movie-ids
   # We use row.names() function
   u1$id<-row.names(u1)
   # Now access movie ratings in column 1 of u1
   x= u1[u1$id==movieid,1]
   # print(u)
   # print(length(x))
   # If no ratings were found, assign 0. You could also
   #   assign user-average
   if (length(x)==0)
   {
     ratings[u] <- 0
   }
   else
   {
     ratings[u] <-x
   }
 
}
length(ratings)
tx<-cbind(test[,1],round(ratings))
# Write to a csv file: submitfile.csv in your folder
write.table(tx,file="submitfile.csv",row.names=FALSE,col.names=FALSE,sep=',')
# Submit now this csv file to kaggle
########################################


























