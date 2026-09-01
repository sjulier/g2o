// Tests the serialization of the level and the robust kernel of an edge.

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "allocate_optimizer.h"
#include "g2o/core/factory.h"
#include "g2o/core/optimizable_graph.h"
#include "g2o/core/robust_kernel.h"
#include "g2o/core/robust_kernel_factory.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/stuff/string_tools.h"
#include "g2o/types/slam2d/types_slam2d.h"  // IWYU pragma: keep
#include "gmock/gmock.h"
#include "gtest/gtest.h"

G2O_USE_TYPE_GROUP(slam2d);

using namespace testing;

namespace {

//! a graph without the optional records, as written by earlier versions of g2o
const char* kLegacyGraph =
    "VERTEX_SE2 0 0 0 0\n"
    "VERTEX_SE2 1 1 0 0\n"
    "FIX 0\n"
    "EDGE_SE2 0 1 1 0 0 1 0 0 1 0 1\n";

//! recover the payload of the lines starting with the given token
std::vector<std::string> parseLines(const std::string& graphData,
                                    const std::string& token) {
  std::vector<std::string> result;
  std::istringstream is(graphData);
  std::stringstream currentLine;
  while (g2o::readLine(is, currentLine) >= 0) {
    std::string firstToken;
    currentLine >> firstToken;
    if (firstToken != token) continue;
    std::string payload;
    std::getline(currentLine, payload);
    result.push_back(g2o::trim(payload));
  }
  return result;
}

}  // namespace

/**
 * Fixture with a small graph of three vertices connected by three edges, all
 * of them on the default level and without a robust kernel.
 */
class IoEdgeAttributes : public Test {
 protected:
  void SetUp() override {
    optimizer.reset(g2o::internal::createOptimizerForTests());

    for (int i = 0; i < numVertices; ++i) {
      g2o::VertexSE2* v = new g2o::VertexSE2;
      v->setEstimate(g2o::SE2());
      v->setId(i);
      v->setFixed(i == 0);  // fix the first vertex
      optimizer->addVertex(v);
    }

    for (int i = 0; i < numVertices; ++i) {
      g2o::EdgeSE2* e = new g2o::EdgeSE2;
      e->vertices()[0] = optimizer->vertex((i + 0) % numVertices);
      e->vertices()[1] = optimizer->vertex((i + 1) % numVertices);
      e->setMeasurement(g2o::SE2(1., 0., 0.));
      e->setInformation(g2o::EdgeSE2::InformationType::Identity());
      optimizer->addEdge(e);
    }
  }

  //! the edge of the graph connecting the vertices with the given IDs
  g2o::OptimizableGraph::Edge* edge(int idFrom, int idTo) const {
    for (const auto& elem : optimizer->edges()) {
      if (elem->vertex(0)->id() == idFrom && elem->vertex(1)->id() == idTo)
        return static_cast<g2o::OptimizableGraph::Edge*>(elem);
    }
    return nullptr;
  }

  //! the tag under which the robust kernel of an edge is registered
  static std::string kernelTag(const g2o::OptimizableGraph::Edge* e) {
    const g2o::RobustKernel* kernel = e->robustKernel();
    if (!kernel) return "";
    return g2o::RobustKernelFactory::instance()->tag(kernel);
  }

  std::unique_ptr<g2o::SparseOptimizer> optimizer;
  int numVertices = 3;
};

TEST_F(IoEdgeAttributes, DefaultsAreNotSaved) {
  std::stringstream graphData;
  ASSERT_TRUE(optimizer->save(graphData));

  // files which can be read by earlier versions of g2o stay unchanged
  EXPECT_THAT(parseLines(graphData.str(), "LEVEL"), IsEmpty());
  EXPECT_THAT(parseLines(graphData.str(), "ROBUSTKERNEL"), IsEmpty());
}

TEST_F(IoEdgeAttributes, SaveRobustKernel) {
  g2o::RobustKernelHuber* kernel = new g2o::RobustKernelHuber;
  kernel->setDelta(1.5);
  edge(0, 1)->setRobustKernel(kernel);

  std::stringstream graphData;
  ASSERT_TRUE(optimizer->save(graphData));

  EXPECT_THAT(parseLines(graphData.str(), "ROBUSTKERNEL"),
              ElementsAre("Huber 1.5"));
}

TEST_F(IoEdgeAttributes, SaveLevel) {
  edge(0, 1)->setLevel(2);

  std::stringstream graphData;
  ASSERT_TRUE(optimizer->save(graphData, 2));

  EXPECT_THAT(parseLines(graphData.str(), "LEVEL"), ElementsAre("2"));
}

TEST_F(IoEdgeAttributes, LoadRobustKernel) {
  g2o::RobustKernelHuber* huber = new g2o::RobustKernelHuber;
  huber->setDelta(1.5);
  edge(0, 1)->setRobustKernel(huber);

  g2o::RobustKernelCauchy* cauchy = new g2o::RobustKernelCauchy;
  cauchy->setDelta(0.25);
  edge(1, 2)->setRobustKernel(cauchy);

  std::stringstream graphData;
  ASSERT_TRUE(optimizer->save(graphData));
  optimizer->clear();
  ASSERT_TRUE(optimizer->load(graphData));
  ASSERT_THAT(optimizer->edges(), SizeIs(numVertices));

  EXPECT_THAT(kernelTag(edge(0, 1)), Eq("Huber"));
  EXPECT_THAT(edge(0, 1)->robustKernel()->delta(), DoubleEq(1.5));
  EXPECT_THAT(kernelTag(edge(1, 2)), Eq("Cauchy"));
  EXPECT_THAT(edge(1, 2)->robustKernel()->delta(), DoubleEq(0.25));
  // the third edge does not have a kernel and must not have gotten one
  EXPECT_THAT(edge(2, 0)->robustKernel(), IsNull());
}

TEST_F(IoEdgeAttributes, LoadLevel) {
  edge(1, 2)->setLevel(3);

  std::stringstream graphData;
  ASSERT_TRUE(optimizer->save(graphData, 3));
  optimizer->clear();
  ASSERT_TRUE(optimizer->load(graphData));

  ASSERT_THAT(optimizer->edges(), SizeIs(1));
  EXPECT_THAT(edge(1, 2)->level(), Eq(3));
}

TEST_F(IoEdgeAttributes, LoadLevelAndRobustKernelTogether) {
  g2o::RobustKernelDCS* kernel = new g2o::RobustKernelDCS;
  kernel->setDelta(2.5);
  edge(0, 1)->setLevel(1);
  edge(0, 1)->setRobustKernel(kernel);

  std::stringstream graphData;
  ASSERT_TRUE(optimizer->save(graphData, 1));
  optimizer->clear();
  ASSERT_TRUE(optimizer->load(graphData));

  ASSERT_THAT(optimizer->edges(), SizeIs(1));
  EXPECT_THAT(edge(0, 1)->level(), Eq(1));
  EXPECT_THAT(kernelTag(edge(0, 1)), Eq("DCS"));
  EXPECT_THAT(edge(0, 1)->robustKernel()->delta(), DoubleEq(2.5));
}

TEST_F(IoEdgeAttributes, LoadLegacyGraph) {
  optimizer->clear();

  std::stringstream graphData(kLegacyGraph);
  ASSERT_TRUE(optimizer->load(graphData));

  ASSERT_THAT(optimizer->edges(), SizeIs(1));
  // an edge of a file without the records defaults to level zero and to not
  // having a robust kernel
  EXPECT_THAT(edge(0, 1)->level(), Eq(0));
  EXPECT_THAT(edge(0, 1)->robustKernel(), IsNull());
}

TEST_F(IoEdgeAttributes, LoadUnknownRobustKernel) {
  optimizer->clear();

  std::stringstream graphData(std::string(kLegacyGraph) +
                              "ROBUSTKERNEL NoSuchKernel 1.5\n");
  ASSERT_TRUE(optimizer->load(graphData));

  ASSERT_THAT(optimizer->edges(), SizeIs(1));
  EXPECT_THAT(edge(0, 1)->robustKernel(), IsNull());
}

TEST_F(IoEdgeAttributes, LoadRecordsWithoutAnEdge) {
  optimizer->clear();

  // the records are ignored, the remainder of the file is still read
  std::stringstream graphData("LEVEL 2\nROBUSTKERNEL Huber 1.5\n" +
                              std::string(kLegacyGraph));
  ASSERT_TRUE(optimizer->load(graphData));

  ASSERT_THAT(optimizer->edges(), SizeIs(1));
  EXPECT_THAT(edge(0, 1)->level(), Eq(0));
  EXPECT_THAT(edge(0, 1)->robustKernel(), IsNull());
}
