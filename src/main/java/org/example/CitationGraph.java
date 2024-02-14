package org.example;

import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.sql.*;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.expressions.WindowSpec;

import java.util.*;

import static org.apache.spark.sql.functions.*;

public class CitationGraph {
    public static void main(String[] args) {
        if (args.length < 2) {
            System.err.println("Usage: CitationGraph <citations_file> <publications_file>");
            System.exit(1);
        }
        SparkSession spark = SparkSession.builder().config("spark.driver.memory", "10g")
                .appName("CitationGraph")
                .master("local[*]")
                .getOrCreate();

        try {
            // Use SparkSession to read text file into a Dataset
            Dataset<Row> citationsDataset = spark.read().text(args[0]).repartition(5);

            Dataset<Row> cleanedupCitationsDataset = citationsDataset.select(
                    split(col("value"), "\t").getItem(0).as("paper_id"),
                    split(col("value"), "\t").getItem(1).as("cited_paper_id"));


            Dataset<Row> publicationDatesDataset = spark.read().text(args[1]).repartition(5);
            Dataset <Row> cleanedupPublicationDataset = publicationDatesDataset.select(
                    split(col("value"), "\t").getItem(0).as("paper_id"),
                    split(col("value"), "\t").getItem(1).as("published_date"));

            Dataset <Row> uniquePublicationsPerYear = computeUniquePublicationPerYear(cleanedupPublicationDataset);
            computeCumulativeNodesPerYear(uniquePublicationsPerYear);
            Dataset <Row> citationsPerYear = computeCitationEdgesPerYear(cleanedupCitationsDataset, cleanedupPublicationDataset);
            computeCumulativeEdgesPerYear(citationsPerYear);
            computeOverallDiameter(citationsDataset, cleanedupPublicationDataset);

        }catch(Exception ex) {
            ex.printStackTrace();
        }
    }

    private static void computeOverallDiameter(Dataset<Row> cleanedupPublicationDataset, Dataset<Row> citationsDataset) {
        Dataset<Row> joinedDataset = citationsDataset.join(cleanedupPublicationDataset, "paper_id");
        joinedDataset = joinedDataset.withColumn("published_year", substring(col("published_date"), 1, 4))
                .filter(col("published_year").lt(2003)
                        .or(col("published_year").equalTo(2003)
                                .and(substring(col("published_date"), 6, 7).leq("04"))));
        //joinedDataset.show();

        Dataset<Row> uniqueCitationsPerYear = joinedDataset.groupBy("published_year").agg(countDistinct("cited_paper_id").alias("unique_citation_edges"));

        Dataset<Citation> citations = citationsDataset.map((MapFunction<Row, Citation>) row -> {
            String[] parts = row.getString(0).split("\\s+");
            int citingPaper = Integer.parseInt(parts[0]);
            int citedPaper = Integer.parseInt(parts[1]);
            return new Citation(citingPaper, citedPaper);
        }, Encoders.bean(Citation.class));

        // Group by the citing paper to get a list of cited papers for each citing paper
        RelationalGroupedDataset groupedCitingToCited = citations.groupBy("citingPaper");

        // Aggregate to get a list of cited papers for each citing paper
        Dataset<Row> aggregatedCitingToCited = groupedCitingToCited.agg(functions.collect_list("citedPaper").alias("citedPapers"));

        // Convert the result to a map
        List<Row> rows = aggregatedCitingToCited.collectAsList();
        //System.out.println(rows.size());
        Map<Integer, List<Integer>> citingToCitedMap = new java.util.HashMap<>();
        for (Row row : rows) {
            int citingPaper = row.getInt(0);
            List<Integer> citedPapers = row.getList(1);
            citingToCitedMap.put(citingPaper, citedPapers);
        }
        int numConnectedPairs = calculateConnectedPairs(citingToCitedMap, 15);
        calculateHopPlot(numConnectedPairs, citingToCitedMap);
        System.out.println("done / number of connected pairs "+ numConnectedPairs);
    }

    private static Integer calculateConnectedPairs (Map <Integer, List <Integer>> graph, Integer distanceThreshold) {
        //Set<Pair<Integer, Integer>> connectedPairs = new HashSet<>();
        int counter = 0;

        for (Map.Entry<Integer, List<Integer>> node : graph.entrySet() ) {
            int currentNode = node.getKey();
            Map <Integer, Integer> distances = bfsShortestPath(graph, currentNode);
            for (Map.Entry <Integer, Integer> dist : distances.entrySet()) {
                int targetNode = dist.getKey();
                int distance = dist.getValue();

                if (distance <= distanceThreshold){
                    if (currentNode != targetNode) {
                        //connectedPairs.add(new Pair<>(currentNode, targetNode));
                       counter++;
                    }
                }
            }
        }
        //System.out.println("Connected pairs "+ connectedPairs);
       // return connectedPairs.size();
        return counter;
    }

    private static Map<Integer, Integer> bfsShortestPath (Map <Integer, List <Integer>> graph, Integer source) {
        Map <Integer, Integer> distances = new HashMap<>();
        Set <Integer> visited = new HashSet<>();
        Queue <Pair<Integer, Integer>> queue = new LinkedList<>();

        queue.add(new Pair<>(source, 0));

        while (!queue.isEmpty()) {
            Pair <Integer, Integer> currentNode = queue.poll();
            int node = currentNode.getFirst();
            int distance = currentNode.getSecond();

            if (!visited.contains(node)) {
                visited.add(node);
                distances.put(node, distance);
                for (int neighbor : graph.getOrDefault(node, new ArrayList<>())) {
                    queue.add(new Pair<>(neighbor, distance + 1));
                }
            }
        }
        return distances;

    }

    private static Set<Pair <Integer, Double>> calculateHopPlot (Integer totalPairs, Map <Integer, List <Integer>> graph) {
        //Set <Pair <Integer, Integer>> discoveredPairs = new HashSet<>();
        int discoveredPairsCounter = 0;
        Double coverageRate = 0.0;
        int maxhop = 15;
        Set <Pair <Integer, Double>> hopCoverage = new HashSet<>();
        boolean diameterFound = false;

        for (int hop = 1; hop <= maxhop; hop++) {
            if (!diameterFound) {
                for (Map.Entry <Integer, List <Integer>> currentNodeInfo: graph.entrySet()) {
                    int currentNode = currentNodeInfo.getKey();
                    Map<Integer, Integer> currentNodeDistanceToOtherNodes = bfsShortestPath(graph, currentNode);

                    Map<Integer, Integer> filteredDistances = new HashMap<>();
                    for (Map.Entry<Integer, Integer> distanceToOthers : currentNodeDistanceToOtherNodes.entrySet()) {
                        if (distanceToOthers.getValue() == hop) {
                            filteredDistances.put(distanceToOthers.getKey(), distanceToOthers.getValue());
                        }
                    }
                    //neighbors within given hop for an individual node
                    for (Map.Entry<Integer, Integer> distanceEntry : filteredDistances.entrySet()) {
                        int targetNode = distanceEntry.getKey();
                        int distance = distanceEntry.getValue();

                        if (distance <= maxhop) {
                            if (currentNode != targetNode) {
                                //discoveredPairs.add(new Pair<>(currentNode, targetNode));
                                discoveredPairsCounter++;
                            }
                        }
                        //coverageRate = (discoveredPairs.size() / (double)totalPairs) * 100;
                        coverageRate = (discoveredPairsCounter / (double)totalPairs) * 100;
                        //System.out.printf("\ncoverage rate for k = %d is %.2f" , distance, coverageRate);
                        if (coverageRate >= 90) {
                            System.out.println(" diameter " + distance);
                            hopCoverage.add(new Pair<>(hop, coverageRate));
                            diameterFound = true;
                            break;
                        }
                    }
                    if (diameterFound) break;
                }
                //add coverage
                hopCoverage.add(new Pair<>(hop, coverageRate));
            }

        }
        System.out.println(hopCoverage);
        return hopCoverage;
    }

    private static Dataset <Row> computeUniquePublicationPerYear(Dataset <Row> cleanedupPublicationDataset) {
        Dataset <Row> parsedDatesDataset = cleanedupPublicationDataset.withColumn("published_year", substring(col("published_date"), 1, 4))
                .filter(col("published_year").lt(2003)
                        .or(col("published_year").equalTo(2003)
                                .and(substring(col("published_date"), 6, 7).leq("04"))));
        //parsedDatesDataset.show();

        Dataset<Row> uniquePapersPerYear = parsedDatesDataset.groupBy("published_year").agg(countDistinct("paper_id").alias("unique_papers"));
        System.out.println("****Unique publication nodes year****");
        uniquePapersPerYear.show();
        return uniquePapersPerYear;
    }

    private static Dataset <Row> computeCitationEdgesPerYear(Dataset<Row> citationsDataset, Dataset <Row> cleanedupPublicationDataset) {
        Dataset<Row> joinedDataset = citationsDataset.join(cleanedupPublicationDataset, "paper_id");
        joinedDataset = joinedDataset.withColumn("published_year", substring(col("published_date"), 1, 4))
                .filter(col("published_year").lt(2003)
                        .or(col("published_year").equalTo(2003)
                                .and(substring(col("published_date"), 6, 7).leq("04"))));
        //joinedDataset.show();

        Dataset<Row> uniqueCitationsPerYear = joinedDataset.groupBy("published_year").agg(countDistinct("cited_paper_id").alias("unique_citation_edges"));
        System.out.println("****Unique citation edges per year****");
        uniqueCitationsPerYear.show();
        return uniqueCitationsPerYear;
    }

    private static void computeCumulativeNodesPerYear(Dataset<Row> uniquePublicationsPerYear) {
        // Calculate the cumulative sum of the number of unique publications per year
        WindowSpec windowSpec = Window.orderBy("published_year").rowsBetween(Window.unboundedPreceding(), Window.currentRow());
        Dataset<Row> cumulativeSumPerYear = uniquePublicationsPerYear
                .filter(col("published_year").between("1992", "2004"))
                .withColumn("End of year (t)", expr("CAST(published_year AS INT) + 1"))
                .withColumn("No of nodes at end of year t (n(t))", sum("unique_papers").over(windowSpec));

        // Show the result
        System.out.println("****Cumulative nodes by end of given year****");
        cumulativeSumPerYear.show();
    }

    private static void computeCumulativeEdgesPerYear(Dataset<Row> citationsPerYear) {
        // Calculate the cumulative sum of the number of unique publications per year
        WindowSpec windowSpec = Window.orderBy("published_year").rowsBetween(Window.unboundedPreceding(), Window.currentRow());
        Dataset<Row> cumulativeSumPerYear = citationsPerYear
                .filter(col("published_year").between("1992", "2004"))
                .withColumn("End of year (t)", expr("CAST(published_year AS INT) + 1"))
                .withColumn("No of edges at end of year t (e(t))", sum("unique_citation_edges").over(windowSpec));

        // Show the result
        System.out.println("****Cumulative edges by end of given year****");
        cumulativeSumPerYear.show();
    }

    public static void computeCitationGraph(Dataset <Row> cleanedUpCitationsDataset) {
        Dataset<Row> graph = cleanedUpCitationsDataset.toDF("paper_id", "cited_paper_id");
        //graph.show();
        Dataset<Row> outDegree = graph.groupBy("paper_id").count().withColumnRenamed("count", "out_degree");
        //outDegree.show();
        Dataset<Row> inDegree = graph.groupBy("cited_paper_id").count().withColumnRenamed("count", "in_degree");
        //inDegree.show();

        Dataset<Row> degreeInfo = outDegree.join(inDegree, graph.col("cited_paper_id").equalTo(inDegree.col("cited_paper_id")), "outer")
                .select(outDegree.col("paper_id"), outDegree.col("out_degree"), inDegree.col("in_degree"))
                .na().fill(0);

        degreeInfo.show();
        // Compute the citation density for each paper
        //Dataset<Row> citationDensityGraph = degreeInfo.withColumn("citation_density", inDegree.col("in_degree").divide(outDegree.col("out_degree")));

        //citationDensityGraph.show();
    }
    public static class Citation {
        private int citingPaper;
        private int citedPaper;

        private int publicationYear;

        public Citation() {}

        public Citation(int citingPaper, int citedPaper, int publicationYear) {
            this.citingPaper = citingPaper;
            this.citedPaper = citedPaper;
            this.publicationYear = publicationYear;
        }
    }
}