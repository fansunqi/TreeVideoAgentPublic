from util import get_duration, get_segment_id
import pdb

# 视频切片作为树节点
class VideoSeg:
    def __init__(self, start, end, segment_id=None, description=None):
        """
        初始化 VideoSeg 对象
        
        :param start: int, 视频切片的起始帧数
        :param end: int, 视频切片的结尾帧数
        :param segment_id: int, LLM 标记的视频切片的 ID
        :param description: str or None, 视频切片的描述（默认为 None）
        """
        self.start = start              # 起始帧数
        self.end = end                  # 结束帧数

        self.segment_id = segment_id    # 无实际作用
        self.description = description  # 无实际作用

    def __repr__(self):
        """
        返回视频切片的简要字符串表示
        """
        return f"VideoSeg(start={self.start}, end={self.end}, segment_id={self.segment_id}, description={self.description})"
    
    def __eq__(self, other):
        """
        判断两个 VideoSeg 实例是否相等
        """
        if isinstance(other, VideoSeg):
            return self.start == other.start and self.end == other.end
        return False


def extract_videoseg_from_descriptions(descriptions):
    """
    根据描述列表提取视频切片实例
    
    :param descriptions: List of dictionaries, 每个字典包含 'segment_id', 'duration', 'description' 或 'display_description'
    :return: List of VideoSeg instances
    """
    video_segments = []
    
    for description in descriptions:

        duration = get_duration(description)

        try:
            start, end = map(int, duration.split('-'))  # 解析 'start-end' 格式
        except ValueError: 
            print(f"ValueError -- extract_videoseg_from_descriptions -- duration:{duration}")
            # duration 只有一个数
            start = int(duration)
            end = int(duration)
            # pdb.set_trace()
        # except AttributeError:  # AttributeError: 'NoneType' object(duration) has no attribute 'split'
        #     print(f"AttributeError -- extract_videoseg_from_descriptions -- description: {description}")
        #     continue
        except:  # AttributeError: 'NoneType' object(duration) has no attribute 'split'
            print(f"Error -- extract_videoseg_from_descriptions -- description: {description}")
            continue


        segment_id = get_segment_id(description)
        
        # 获取描述（优先使用 'description'，如果没有则设置为 None)
        description = description.get('description', None)
        
        # 创建 VideoSeg 实例
        video_seg = VideoSeg(start, end, segment_id, description)

        if video_seg not in video_segments:  # 防止重复
            video_segments.append(video_seg)
    
    return video_segments


# TODO: LLM 直接想象中间哪一帧比较重要，而不是二分插值
# TODO: 用 CLIP 插帧
# TODO: 间隔比较大就多插帧，反之，少插帧
def split_and_reconnect_segments(selected_video_segments, video_segments, for_seg_not_interested, num_frames):

    # TODO 确保 video_segments 一定是有序，无重复的

    new_segments = []

    if for_seg_not_interested == "prune":
    
        # 对每个选中的视频切片进行二分
        for segment in selected_video_segments:
            
            if segment.start >= segment.end - 1:
                # 如果 seg 只有一或两张图片，就不可分了
                new_segments.append(segment)
            else:
                mid_point = (segment.start + segment.end) // 2  # 计算中点

                # 创建两个新的 VideoSeg 实例
                first_half = VideoSeg(start=segment.start, end=mid_point)
                second_half = VideoSeg(start=mid_point, end=segment.end)
                
                # 将新生成的两个切片添加到结果列表中
                new_segments.append(first_half)
                new_segments.append(second_half)
    
    elif for_seg_not_interested == "retain":

        for segment in video_segments:

            if segment in selected_video_segments:
                if segment.start >= segment.end - 1:
                    # 如果 seg 只有一张图片，就不可分了
                    new_segments.append(segment)
                else:
                    mid_point = (segment.start + segment.end) // 2  # 计算中点

                    # 创建两个新的 VideoSeg 实例
                    first_half = VideoSeg(start=segment.start, end=mid_point)
                    second_half = VideoSeg(start=mid_point, end=segment.end)
                    
                    # 将新生成的两个切片添加到结果列表中
                    new_segments.append(first_half)
                    new_segments.append(second_half)
            else:
                new_segments.append(segment)

    elif for_seg_not_interested == "merge":
        
        for i, segment in enumerate(selected_video_segments):

            if i == 0:
                # 把头部那一段连上
                if segment.start != 1:
                    video_start_seg = VideoSeg(start=1, end=segment.start)
                    new_segments.append(video_start_seg)

            # 把之前缺失的若干段 merge 成一个新节点
            if i != 0 and segment.start != new_segments[-1].end:
                video_merged_seg = VideoSeg(start=new_segments[-1].end, end=segment.start)
                new_segments.append(video_merged_seg)


            if segment.start >= segment.end - 1:
                # 如果 seg 只有一张图片，就不可分了
                new_segments.append(segment)
            else:
                mid_point = (segment.start + segment.end) // 2  # 计算中点

                # 创建两个新的 VideoSeg 实例
                first_half = VideoSeg(start=segment.start, end=mid_point)
                second_half = VideoSeg(start=mid_point, end=segment.end)
                
                # 将新生成的两个切片添加到结果列表中
                new_segments.append(first_half)
                new_segments.append(second_half)

            if i == len(selected_video_segments) - 1:
                # 尾部一段也连上
                if segment.start != 180:
                    video_start_seg = VideoSeg(start=segment.end, end=num_frames)
                    new_segments.append(video_start_seg)
            
    else:
        raise KeyError
    
    return new_segments