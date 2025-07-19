import { load } from "@tensorflow-models/universal-sentence-encoder";
import { tensor2d } from "@tensorflow/tfjs-node";
import { TDataset } from "../types.js";

export const inputEncoder = async (data: string[]) => {
  try {
    const model = await load();
    const tensor = await model.embed(data);
    const dataArray = tensor.array();
    return dataArray;
  } catch (err) {
    console.error("Fit Error:", err);
    return null;
  }
};

export const outputEncoder = async (data: TDataset[]) => {
  return tensor2d(data.map(ele => {
    const label = ele.label;
    return [
      label === 't_skill' ? 1 : 0,
      label === 'o_skill' ? 1 : 0,
    ]
  }));
}