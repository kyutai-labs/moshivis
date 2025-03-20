import { useServerInfo } from "../../hooks/useServerInfo";

function pretty_format(num: number): number {
  return Math.round((num + Number.EPSILON) * 100) / 100
}

export const ServerInfo = (props: { setFileName: Function }) => {
  const { serverInfo } = useServerInfo();
  if (!serverInfo) {
    return null;
  }
  props.setFileName(serverInfo.base_filename);
  return (
    <div className="p-2 pt-4 self-center flex flex-col text-white border-2 border-white break-words">
      Our server is running on the following configuration:
      <div>Image resolution: {serverInfo.image_resolution} px</div>
      <div>Text temperature: {pretty_format(serverInfo.text_temperature)}</div>
      <div>Text topk: {serverInfo.text_topk}</div>
      <div>Temperature gating: {pretty_format(serverInfo.text_temperature_gating_influence)}</div>
      <div>Audio temperature: {pretty_format(serverInfo.audio_temperature)}</div>
      <div>Audio topk: {serverInfo.audio_topk}</div>
      <div>Pad mult: {serverInfo.pad_mult}</div>
      <div>Repeat penalty last N: {serverInfo.repetition_penalty_context}</div>
      <div>Repeat penalty: {serverInfo.repetition_penalty}</div>
      <div>LM model file: {serverInfo.lm_model_file}</div>
      <div>Instance name: {serverInfo.instance_name}</div>
    </div>
  );
};
