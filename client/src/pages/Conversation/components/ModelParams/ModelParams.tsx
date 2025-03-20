import { FC, RefObject } from "react";
import { useModelParams } from "../../hooks/useModelParams";
import { Button } from "../../../../components/Button/Button";

type ModelParamsProps = {
  isConnected: boolean;
  modal?: RefObject<HTMLDialogElement>,
} & ReturnType<typeof useModelParams>;
export const ModelParams: FC<ModelParamsProps> = ({
  textTemperature,
  textTopk,
  audioTemperature,
  audioTopk,
  padMult,
  repetitionPenalty,
  repetitionPenaltyContext,
  imageResolution,
  gateInfluence,
  displayColor,
  centerCrop,
  setTextTemperature,
  setTextTopk,
  setAudioTemperature,
  setAudioTopk,
  setPadMult,
  setRepetitionPenalty,
  setRepetitionPenaltyContext,
  setImageResolution,
  setGateInfluence,
  setDisplayColor,
  setCenterCrop,
  resetParams,
  isConnected,
  modal,
}) => {
  return (
    <div className=" p-2 mt-6 self-center flex flex-col text-white items-center text-center">
      {!isConnected && <span className="text-xs italic mb-3">Hover on each element to display a helpful tooltip</span>}
      <table>
        <tbody>
          <tr title="Sampling temperature for Moshi's text tokens ('inner monologue')">
            <td>Text temperature:</td>
            <td className="w-12 text-center">{textTemperature}</td>
            <td className="p-2"><input className="range align-middle" disabled={isConnected} type="range" id="text-temperature" name="text-temperature" step="0.01" min="0.2" max="1.2" value={textTemperature} onChange={e => setTextTemperature(parseFloat(e.target.value))} /></td>
          </tr>
          <tr>
            <td title="Sampling top-k for Moshi's text tokens ('inner monologue')">Text topk:</td>
            <td className="w-12 text-center">{textTopk}</td>
            <td className="p-2"><input className="range align-middle" disabled={isConnected} type="range" id="text-topk" name="text-topk" step="1" min="10" max="500" value={textTopk} onChange={e => setTextTopk(parseInt(e.target.value))} /></td>
          </tr>
          <tr title="Sampling temperature for Moshi's audio tokens">
            <td>Audio temperature:</td>
            <td className="w-12 text-center">{audioTemperature}</td>
            <td className="p-2"><input className="range align-middle" disabled={isConnected} type="range" id="audio-temperature" name="audio-temperature" step="0.01" min="0.2" max="1.2" value={audioTemperature} onChange={e => setAudioTemperature(parseFloat(e.target.value))} /></td>
          </tr>
          <tr title="Sampling top-k for Moshi's audio tokens">
            <td>Audio topk:</td>
            <td className="w-12 text-center">{audioTopk}</td>
            <td className="p-2"><input className="range align-middle" disabled={isConnected} type="range" id="audio-topk" name="audio-topk" step="1" min="10" max="500" value={audioTopk} onChange={e => setAudioTopk(parseInt(e.target.value))} /></td>
          </tr>
          <tr title="Up/Down weight the text padding token (lower values make Moshi more reactive)">
            <td>Padding multiplier:</td>
            <td className="w-12 text-center">{padMult}</td>
            <td className="p-2"><input className="range align-middle" disabled={isConnected} type="range" id="audio-pad-mult" name="audio-pad-mult" step="0.05" min="-4" max="4" value={padMult} onChange={e => setPadMult(parseFloat(e.target.value))} /></td>
          </tr>
          <tr title="Up/Down weight repeated tokens (higher values enforce fewer repetitions)">
            <td>Repeat penalty:</td>
            <td className="w-12 text-center">{repetitionPenalty}</td>
            <td className="p-2"><input className="range align-middle" disabled={isConnected} type="range" id="repetition-penalty" name="repetition-penalty" step="0.01" min="1" max="2" value={repetitionPenalty} onChange={e => setRepetitionPenalty(parseFloat(e.target.value))} /></td>
          </tr>
          <tr title="Which horizon to consider for the repeat penalty">
            <td>Repeat penalty last N:</td>
            <td className="w-12 text-center">{repetitionPenaltyContext}</td>
            <td className="p-2"><input className="range align-middle" disabled={isConnected} type="range" id="repetition-penalty-context" name="repetition-penalty-context" step="1" min="0" max="200" value={repetitionPenaltyContext} onChange={e => setRepetitionPenaltyContext(parseFloat(e.target.value))} /></td>
          </tr>
          <tr title="Input image resolution in pixels (the largest side will be resized to the given size)">
            <td>Image max-side (px):</td>
            <td className="w-12 text-center">{imageResolution}</td>
            <td className="p-2"><input className="range align-middle" disabled={isConnected} type="range" id="image-resolution" name="image-resolution" step="16" min="64" max="512" value={imageResolution} onChange={e => setImageResolution(parseFloat(e.target.value))} /></td>
          </tr>
          <tr title="Whether to center crop the image to square or keep its original aspect ratio">
            <td>Center Crop:</td>
            <td className="w-12 text-center">{centerCrop ? '✔️' : '✖️'}</td>
            <td className="p-2"><input className="range align-middle" disabled={isConnected} type="range" id="center-crop" name="center-crop" step="1" min="0" max="1" value={centerCrop ? 1 : 0} onChange={e => setCenterCrop((parseFloat(e.target.value) == 1) ? true : false)} /></td>
          </tr>
          <tr title="Whether to display MoshiVis's gates' outputs via the text color (orange indicates more image relevance; green, more general knowledge)">
            <td>Display Gating:</td>
            <td className="w-12 text-center">{displayColor ? '✔️' : '✖️'}</td>
            <td className="p-2"><input className="range align-middle" disabled={isConnected} type="range" id="display-color" name="display-color" step="1" min="0" max="1" value={displayColor ? 1 : 0} onChange={e => setDisplayColor((parseFloat(e.target.value) == 1) ? true : false)} /></td>
          </tr>
          <tr title="Whether and how much to rescale the text temperature based on MoshiVis's gates' outputs (higher value = text temperature will be lowered when the gates are active, i.e. when the tokens are image relevant)">
            <td>Temperature Gating:</td>
            <td className="w-12 text-center">{gateInfluence}</td>
            <td className="p-2"><input className="range align-middle" disabled={isConnected} type="range" id="gate-influence" name="gate-influence" step="0.01" min="0.0" max="0.99" value={gateInfluence} onChange={e => setGateInfluence(parseFloat(e.target.value))} /></td>
          </tr>
        </tbody>
      </table>
      <div>
        {!isConnected && <Button onClick={resetParams} className="mt-6 mr-4">Reset</Button>}
        {!isConnected && <Button onClick={() => modal?.current?.close()} className="mt-6 ml-4">Ok</Button>}
      </div>
    </div >
  )
};
