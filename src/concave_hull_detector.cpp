/*

Most part of this code is written by LiLaBoC, and M.Tsuru (AIST, Japan) modified it for visualization.
protected by MIT License.
For commercial use, I hope you contact us, to know who/how-to use this code in our human-robot society,
and to help you with giving much more technical advices.

*/


#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/extract_indices.h>
// #include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/surface/concave_hull.h>
#include <visualization_msgs/Marker.h>

class Concave_Hull_Detector
{
private:
  /*node handle*/
  ros::NodeHandle nh;
  ros::NodeHandle nhPrivate;
  /*subscribe*/
  ros::Subscriber sub_pc;
  /*publishe*/
  ros::Publisher pub_pc;
  ros::Publisher pub_bbx;
  sensor_msgs::PointCloud2 latest_msg_;
  /*pcl objects*/
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud{new pcl::PointCloud<pcl::PointXYZ>};
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters;
  // pcl::visualization::PCLVisualizer viewer{"Euclidian Clustering"};
  /*parameters*/
  double ratio_depth_tolerance;
  double min_tolerance;
  double max_tolerance;
  int min_cluster_size;
  int max_cluster_size;

public:
  Concave_Hull_Detector();
  void CallbackPC(const sensor_msgs::PointCloud2ConstPtr &msg);
  void Sampling();
  bool remove_plane(const pcl::PointCloud<pcl::PointXYZ>::Ptr &input_cloud, const Eigen::Vector3f &axis = Eigen::Vector3f(0.0, 0.0, 1.0), double plane_thickness = 0.01);
  void Clustering(void);
  void GenConvexHull();
  double ComputeTolerance(const pcl::PointXYZ &point);
  bool CustomCondition(const pcl::PointXYZ &seed_point, const pcl::PointXYZ &candidate_point, float squared_distance);
  void Visualization(void);
  void Publish();
  void Publish_BBX();
};

Concave_Hull_Detector::Concave_Hull_Detector()
    : nhPrivate("~")
{
  sub_pc = nh.subscribe("/camera/depth/color/points", 1, &Concave_Hull_Detector::CallbackPC, this);
  pub_pc = nh.advertise<sensor_msgs::PointCloud2>("/clustered_pc", 1);
  pub_bbx = nh.advertise<visualization_msgs::Marker>("/bounding_box", 1);
  // viewer.setBackgroundColor(1, 1, 1);
  // viewer.addCoordinateSystem(1.0, "axis");
  // viewer.setCameraPosition(0.0, 0.0, 35.0, 0.0, 0.0, 0.0);

  nhPrivate.param("ratio_depth_tolerance", ratio_depth_tolerance, 0.05);
  std::cout << "ratio_depth_tolerance = " << ratio_depth_tolerance << std::endl;
  nhPrivate.param("min_tolerance", min_tolerance, 0.1);
  std::cout << "min_tolerance = " << min_tolerance << std::endl;
  nhPrivate.param("max_tolerance", max_tolerance, 0.5);
  std::cout << "max_tolerance = " << max_tolerance << std::endl;
  nhPrivate.param("min_cluster_size", min_cluster_size, 100);
  std::cout << "min_cluster_size = " << min_cluster_size << std::endl;
}

void Concave_Hull_Detector::CallbackPC(const sensor_msgs::PointCloud2ConstPtr &msg)
{
  /* std::cout << "CALLBACK PC" << std::endl; */
  latest_msg_ = *msg; // keep it in the local member

  /*ROS点群→PCL点群*/
  pcl::fromROSMsg(*msg, *cloud);
  std::cout << "==========" << std::endl;
  std::cout << "cloud->points.size() = " << cloud->points.size() << std::endl;

  max_cluster_size = cloud->points.size();
  clusters.clear();

  Sampling();
  remove_plane(cloud);
  Clustering();
  GenConvexHull();
  // Visualization(); // Let's visualize on RViz
  Publish();
  Publish_BBX();
}

void Concave_Hull_Detector::Sampling(void)
{
  pcl ::VoxelGrid<pcl ::PointXYZ> filter;
  filter.setInputCloud(cloud);
  // set the size of the voxel grid 1x1x1cm
  filter.setLeafSize(0.01f, 0.01f, 0.01f);
  filter.filter(*cloud);
}

bool Concave_Hull_Detector::remove_plane(const pcl::PointCloud<pcl::PointXYZ>::Ptr &input_cloud, const Eigen::Vector3f &axis, double plane_thickness)
{
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  // Create the segmentation object
  pcl::SACSegmentation<pcl::PointXYZ> seg;

  // Optional
  seg.setOptimizeCoefficients(true);
  // Mandatory
  seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setMaxIterations(500);
  seg.setAxis(axis);
  seg.setEpsAngle(0.25);
  seg.setDistanceThreshold(plane_thickness); //0.025 0.018
  seg.setInputCloud(input_cloud);
  seg.segment(*inliers, *coefficients);
  //ROS_INFO("plane size : %d", inliers->indices.size());

  if (inliers->indices.size() < 500)
  {
    ROS_INFO("plane size is not enough large to remove.");
    return false;
  }
  ROS_INFO("found a plane.");
  pcl::ExtractIndices<pcl::PointXYZ> extract;
  extract.setInputCloud(input_cloud);
  extract.setIndices(inliers);
  extract.setNegative(true);
  extract.filter(*input_cloud);

  return true;
}

void Concave_Hull_Detector::Clustering(void)
{
  double time_start = ros::Time::now().toSec();

  /*search config*/
  /*kd-treeクラスを宣言*/
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  /*探索する点群をinput*/
  kdtree.setInputCloud(cloud);
  max_cluster_size = cloud->points.size();
  /*objects*/
  std::vector<pcl::PointIndices> cluster_indices;
  std::vector<bool> processed(cloud->points.size(), false);
  std::vector<int> nn_indices;
  std::vector<float> nn_distances;
  /*clustering*/
  for (size_t i = 0; i < cloud->points.size(); ++i)
  {
    if (processed[i])
      continue; //既に分類されているかチェック
    /*set seed（シード点を設定）*/
    std::vector<int> seed_queue;
    int sq_idx = 0;
    seed_queue.push_back(i);
    processed[i] = true;
    /*clustering*/
    while (sq_idx < seed_queue.size())
    { //探索しきるまでループ
      if (sq_idx % 10000 == 0)
        std::cout << "clustering time [s] = " << ros::Time::now().toSec() - time_start << std::endl;
      /*search*/
      double tolerance = ComputeTolerance(cloud->points[seed_queue[sq_idx]]);
      int ret = kdtree.radiusSearch(cloud->points[seed_queue[sq_idx]], tolerance, nn_indices, nn_distances);
      if (ret == -1)
      {
        PCL_ERROR("[pcl::extractEuclideanClusters] Received error code -1 from radiusSearch\n");
        exit(0);
      }
      /*check*/
      for (size_t j = 0; j < nn_indices.size(); ++j)
      {
        /*//既に分類されているかチェック*/
        if (nn_indices[j] == -1 || processed[nn_indices[j]])
          continue;
        /*カスタム条件でチェック*/
        if (CustomCondition(cloud->points[seed_queue[sq_idx]], cloud->points[nn_indices[j]], nn_distances[j]))
        {
          seed_queue.push_back(nn_indices[j]);
          processed[nn_indices[j]] = true;
        }
      }
      sq_idx++;
    }
    /*judge（クラスタのメンバ数が条件を満たしているか）*/
    if (seed_queue.size() >= min_cluster_size && seed_queue.size() <= max_cluster_size)
    {
      pcl::PointIndices tmp_indices;
      tmp_indices.indices = seed_queue;
      cluster_indices.push_back(tmp_indices);
    }
  }
  std::cout << "cluster_indices.size() = " << cluster_indices.size() << std::endl;
  /*extraction（クラスタごとに点群を分割）*/
  pcl::ExtractIndices<pcl::PointXYZ> ei;
  ei.setInputCloud(cloud);
  ei.setNegative(false);
  for (size_t i = 0; i < cluster_indices.size(); i++)
  {
    /*extract*/
    pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_clustered_points(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointIndices::Ptr tmp_clustered_indices(new pcl::PointIndices);
    *tmp_clustered_indices = cluster_indices[i];
    ei.setIndices(tmp_clustered_indices);
    ei.filter(*tmp_clustered_points);
    /*input*/
    clusters.push_back(tmp_clustered_points);
  }

  std::cout << "clustering time [s] = " << ros::Time::now().toSec() - time_start << std::endl;
}

void Concave_Hull_Detector::GenConvexHull()
{
  for(size_t i=0; i < clusters.size(); i++)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ConcaveHull<pcl::PointXYZ> chull;
    chull.setInputCloud (clusters.at(i));
    chull.setAlpha (0.1);
    chull.reconstruct (*cloud_hull);

    std::cerr << "Concave hull" << i << " has: " << cloud_hull->size ()
            << " data points." << std::endl;
  }
}

double Concave_Hull_Detector::ComputeTolerance(const pcl::PointXYZ &point)
{
  /*センサからの距離（depth）*/
  double depth = sqrt(
      point.x * point.x + point.y * point.y + point.z * point.z);

  double tolerance = ratio_depth_tolerance * depth; //距離に比例
  if (tolerance < min_tolerance)
    tolerance = min_tolerance;
  if (tolerance > max_tolerance)
    tolerance = max_tolerance;

  return tolerance;
}

bool Concave_Hull_Detector::CustomCondition(const pcl::PointXYZ &seed_point, const pcl::PointXYZ &candidate_point, float squared_distance)
{
  return true;
}

void Concave_Hull_Detector::Visualization(void)
{
  /*前ステップの可視化をリセット*/
  // viewer.removeAllPointClouds();

  /*cloud*/
  // viewer.addPointCloud(cloud, "cloud");
  // viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 0.0, "cloud");
  // viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");
  /*clusters*/
  double rgb[3] = {};
  const int channel = 3;                                                     // RGB
  const double step = ceil(pow(clusters.size() + 2, 1.0 / (double)channel)); // exept (000),(111)
  const double max = 1.0;
  /*クラスタをいい感じに色分け*/
  for (size_t i = 0; i < clusters.size(); i++)
  {
    std::string name = "cluster_" + std::to_string(i);
    rgb[0] += 1 / step;
    for (int j = 0; j < channel - 1; j++)
    {
      if (rgb[j] > max)
      {
        rgb[j] -= max + 1 / step;
        rgb[j + 1] += 1 / step;
      }
    }
    // viewer.addPointCloud(clusters[i], name);
    // viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, rgb[0], rgb[1], rgb[2], name);
    // viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, name);
  }
  /*表示の更新*/
  // viewer.spinOnce();
}

void Concave_Hull_Detector::Publish()
{
  pcl::PointCloud<pcl::PointXYZRGB> colored_pc, clustered_pc;
  colored_pc.clear();

  double rgb[3] = {};
  const int channel = 3;                                                     // RGB
  const double step = ceil(pow(clusters.size() + 2, 1.0 / (double)channel)); // exept (000),(111)
  const double max = 1.0;
  /*クラスタをいい感じに色分け*/
  for (size_t i = 0; i < clusters.size(); i++)
  {
    std::string name = "cluster_" + std::to_string(i);
    rgb[0] += 1 / step;
    for (int j = 0; j < channel - 1; j++)
    {
      if (rgb[j] > max)
      {
        rgb[j] -= max + 1 / step;
        rgb[j + 1] += 1 / step;
      }
    }
    pcl::copyPointCloud(*clusters[i], colored_pc);
    for (size_t k = 0; k < colored_pc.size(); k++)
    {
      colored_pc.at(k).r = rgb[0] * 255.0f;
      colored_pc.at(k).g = rgb[1] * 255.0f;
      colored_pc.at(k).b = rgb[2] * 255.0f;
    }
    clustered_pc += colored_pc; // merge
    // viewer.addPointCloud(clusters[i], name);
    // viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, rgb[0], rgb[1], rgb[2], name);
    // viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, name);
  }

  sensor_msgs::PointCloud2 result_msg;
  pcl::toROSMsg(clustered_pc, result_msg);
  result_msg.header = latest_msg_.header;

  pub_pc.publish(result_msg);
}

void Concave_Hull_Detector::Publish_BBX()
{
  visualization_msgs::Marker marker;
  marker.header = latest_msg_.header;
  marker.header.stamp = ros::Time::now();
  marker.ns = "basic_shapes";
  marker.id = 0;

  marker.type = visualization_msgs::Marker::CUBE;
  marker.action = visualization_msgs::Marker::ADD;
  marker.lifetime = ros::Duration();

  marker.scale.x = 1.0;
  marker.scale.y = 1.0;
  marker.scale.z = 1.0;
  marker.pose.position.x = 0;
  marker.pose.position.y = 0;
  marker.pose.position.z = 0;
  marker.pose.orientation.x = 0;
  marker.pose.orientation.y = 0;
  marker.pose.orientation.z = 0;
  marker.pose.orientation.w = 1;
  marker.color.r = 0.0f;
  marker.color.g = 1.0f;
  marker.color.b = 0.0f;
  marker.color.a = 0.5f;
  pub_bbx.publish(marker);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "euclidean_clustering_flexible_tolerance");

  Concave_Hull_Detector euclidean_clustering_flexible_tolerance;

  ros::spin();
}