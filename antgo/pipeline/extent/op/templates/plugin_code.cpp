#include "eagleeye/common/EagleeyeMacro.h"
#include "eagleeye/framework/pipeline/AnyNode.h"
#include "eagleeye/framework/pipeline/SignalFactory.h"
#include "eagleeye/framework/pipeline/AnyPipeline.h"
#include "eagleeye/processnode/Placeholder.h"
${include_list}
#include "${project}_plugin.h"
#include "eagleeye/processnode/NNNode.h"

namespace eagleeye{
using namespace dataflow;

/**
 * @brief register pipeline plugin
 * 
 */
EAGLEEYE_PIPELINE_REGISTER(${project}, ${version}, ${signature});

/**
 * @brief configure pipeline plugin
 * 
 */
EAGLEEYE_BEGIN_PIPELINE_INITIALIZE(${project})
// 1.step build datasource node
std::vector<int> in_port_list = std::vector<int>(${in_port});
std::vector<std::string> in_signal_list = std::vector<std::string>(${in_signal});
for(size_t i=0; i<in_port_list.size(); ++i){
    AnyNode* input_node = NULL;
    std::string type_str = in_signal_list[i];
    if(type_str == "EAGLEEYE_SIGNAL_IMAGE" ||
        type_str == "EAGLEEYE_SIGNAL_RGB_IMAGE" ||
        type_str == "EAGLEEYE_SIGNAL_BGR_IMAGE"){
        input_node = new Placeholder<ImageSignal<Array<unsigned char, 3>>>();        
        if(type_str == "EAGLEEYE_SIGNAL_IMAGE"){
            input_node->getOutputPort(0)->setSignalType(EAGLEEYE_SIGNAL_IMAGE);
        }
        else if(type_str == "EAGLEEYE_SIGNAL_RGB_IMAGE"){
            input_node->getOutputPort(0)->setSignalType(EAGLEEYE_SIGNAL_RGB_IMAGE);
        }
        else{
            input_node->getOutputPort(0)->setSignalType(EAGLEEYE_SIGNAL_BGR_IMAGE);
        }
    }
    else if(type_str == "EAGLEEYE_SIGNAL_RGBA_IMAGE" ||
            type_str == "EAGLEEYE_SIGNAL_BGRA_IMAGE"){
        input_node = new Placeholder<ImageSignal<Array<unsigned char, 4>>>();

        if(type_str == "EAGLEEYE_SIGNAL_RGBA_IMAGE"){
            input_node->getOutputPort(0)->setSignalType(EAGLEEYE_SIGNAL_RGBA_IMAGE);
        }
        else{
            input_node->getOutputPort(0)->setSignalType(EAGLEEYE_SIGNAL_BGRA_IMAGE);
        }
    }
    else if(type_str == "EAGLEEYE_SIGNAL_GRAY_IMAGE" ||
            type_str == "EAGLEEYE_SIGNAL_MASK"){
        input_node = new Placeholder<ImageSignal<unsigned char>>();

        if(type_str == "EAGLEEYE_SIGNAL_GRAY_IMAGE"){
            input_node->getOutputPort(0)->setSignalType(EAGLEEYE_SIGNAL_GRAY_IMAGE);
        }
        else{
            input_node->getOutputPort(0)->setSignalType(EAGLEEYE_SIGNAL_MASK);
        }
    }
    else if(type_str == "EAGLEEYE_SIGNAL_DET" ||
            type_str == "EAGLEEYE_SIGNAL_DET_EXT" ||
            type_str == "EAGLEEYE_SIGNAL_TRACKING" ||
            type_str == "EAGLEEYE_SIGNAL_POS_2D" ||
            type_str == "EAGLEEYE_SIGNAL_POS_3D" ||
            type_str == "EAGLEEYE_SIGNAL_LANDMARK"){
        input_node = new Placeholder<ImageSignal<float>>();
        if(type_str == "EAGLEEYE_SIGNAL_DET"){
            input_node->getOutputPort(0)->setSignalType(EAGLEEYE_SIGNAL_DET);
        }
        else if(type_str == "EAGLEEYE_SIGNAL_DET_EXT"){
            input_node->getOutputPort(0)->setSignalType(EAGLEEYE_SIGNAL_DET_EXT);
        }
        else if(type_str == "EAGLEEYE_SIGNAL_TRACKING"){
            input_node->getOutputPort(0)->setSignalType(EAGLEEYE_SIGNAL_TRACKING);
        }
        else if(type_str == "EAGLEEYE_SIGNAL_POS_2D"){
            input_node->getOutputPort(0)->setSignalType(EAGLEEYE_SIGNAL_POS_2D);
        }
        else if(type_str == "EAGLEEYE_SIGNAL_POS_3D"){
            input_node->getOutputPort(0)->setSignalType(EAGLEEYE_SIGNAL_POS_3D);
        }
        else{
            input_node->getOutputPort(0)->setSignalType(EAGLEEYE_SIGNAL_LANDMARK);
        }
    }
    else if(type_str == "EAGLEEYE_SIGNAL_CLS" ||
            type_str == "EAGLEEYE_SIGNAL_STATE"){
        input_node = new Placeholder<ImageSignal<int>>();
        if(type_str == "EAGLEEYE_SIGNAL_CLS"){
            input_node->getOutputPort(0)->setSignalType(EAGLEEYE_SIGNAL_CLS);
        }
        else{
            input_node->getOutputPort(0)->setSignalType(EAGLEEYE_SIGNAL_STATE);
        }
    }
    else if(type_str == "EAGLEEYE_SIGNAL_SWITCH"){
        input_node = new Placeholder<BooleanSignal>();
        input_node->getOutputPort(0)->setSignalType(EAGLEEYE_SIGNAL_SWITCH);
    }
    else if(type_str == "EAGLEEYE_SIGNAL_RECT" ||
            type_str == "EAGLEEYE_SIGNAL_LINE" ||
            type_str == "EAGLEEYE_SIGNAL_POINT"){
        input_node = new Placeholder<ImageSignal<float>>();
        if(type_str == "EAGLEEYE_SIGNAL_RECT"){
            input_node->getOutputPort(0)->setSignalType(EAGLEEYE_SIGNAL_RECT);
        }
        else if(type_str == "EAGLEEYE_SIGNAL_LINE"){
            input_node->getOutputPort(0)->setSignalType(EAGLEEYE_SIGNAL_LINE);
        }
        else{
            input_node->getOutputPort(0)->setSignalType(EAGLEEYE_SIGNAL_POINT);
        }
    }
    else{
        EAGLEEYE_LOGE("Dont support signal type %s.", type_str.c_str());
    }

    std::string placeholer_i_name = formatString("placeholder_%d", i);
    ${project}->add(input_node, placeholer_i_name.c_str());
}

// 2.step build your algorithm node
NNNode* nnnode = new NNNode();
dataflow::Graph* op_graph = nnnode->getOpGraph();
${op_graph}

nnnode->analyze(${graph_in_ops}, ${graph_out_ops});

std::vector<int> out_port_list = std::vector<int>{${out_port}};
std::vector<std::string> out_signal_list = std::vector<std::string>{${out_signal}};
for(size_t i=0; i<out_port_list.size(); ++i){
    nnnode->makeOutputSignal(out_port_list[i], out_signal_list[i]);
}

// 3.step add all node to pipeline
// 3.1.step add data source node
// 3.2.step add your algorithm node
${project}->add(nnnode, "nnnode");

// 4.step link all node in pipeline
for(size_t i=0; i<nnnode->getOpGraphIn(); ++i){
    std::string placeholer_i_name = formatString("placeholder_%d", i);
    ${project}->bind(placeholer_i_name.c_str(), i, "nnnode", i);
}

EAGLEEYE_END_PIPELINE_INITIALIZE
}