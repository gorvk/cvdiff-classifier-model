import * as tf from "@tensorflow/tfjs-node";
import * as use from "@tensorflow-models/universal-sentence-encoder";
import dataset from "./data/training_data.json" with { type: "json" };

const encodeData = async (data) => {
    try {
        const model = await use.load();
        return await model.embed(
            data.map((ele) => ele.input.toLowerCase())
        );
    } catch (err) {
        console.error("Fit Error:", err);
        return null;
    }
};

const getOutputData = async (data) => {
    return tf.tensor2d(data.map(ele => [
        ele.label === 't_skill' ? 1 : 0,
        ele.label === 'o_skill' ? 1 : 0,
    ]));
}

const createModel = async () => {
    const model = tf.sequential();

    model.add(tf.layers.dense({
        inputShape: [512],
        activation: 'sigmoid',
        units: 2,
    }));

    model.add(tf.layers.dense({
        inputShape: [2],
        activation: 'sigmoid',
        units: 2,
    }));

    model.add(tf.layers.dense({
        inputShape: [2],
        activation: 'sigmoid',
        units: 2,
    }));

    model.compile({
        loss: 'meanSquaredError',
        optimizer: tf.train.adam(.06),
    });

    return model;
}

const trainModel = async (model, trainingData, epochs) => {
    const input = await encodeData(trainingData);
    const output = await getOutputData(trainingData);
    model.fit(input, output, { epochs });
}

const predict = async (model, testingData) => {
    const input = await encodeData(testingData);
    const data = await model.predict(input).argMax(1).data();
    return data;
}

const run = async () => {
    const model = await createModel();
    await trainModel(model, dataset, 14)
    model.save('file://model');
}

run();