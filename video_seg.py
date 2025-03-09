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